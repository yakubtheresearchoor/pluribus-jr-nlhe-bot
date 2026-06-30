use std::path::PathBuf;

use metal::{
    CompileOptions, ComputePipelineState, Device, Function, Library,
    MTLSize, CommandQueue,
};

use super::buffer::MetalBuffer;

/// Error type for Metal operations.
#[derive(Debug)]
pub enum MetalError {
    DeviceNotFound,
    LibraryCompilation(String),
    FunctionNotFound(String),
    PipelineCreation(String),
    ShaderFileNotFound(String),
    /// Buffer allocation failed or would exceed a device / layout
    /// limit. Carries a human-readable capacity breakdown so callers
    /// can report it and fall back (e.g. native → hybrid path).
    CapacityExceeded(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::DeviceNotFound => write!(f, "No Metal device found"),
            MetalError::LibraryCompilation(msg) => write!(f, "Shader compilation failed: {}", msg),
            MetalError::FunctionNotFound(name) => write!(f, "Kernel function '{}' not found", name),
            MetalError::PipelineCreation(msg) => write!(f, "Pipeline creation failed: {}", msg),
            MetalError::ShaderFileNotFound(path) => write!(f, "Shader file not found: {}", path),
            MetalError::CapacityExceeded(msg) => write!(f, "GPU capacity exceeded: {}", msg),
        }
    }
}

impl std::error::Error for MetalError {}

pub type MetalResult<T> = Result<T, MetalError>;

/// Top-level Metal context wrapping device, queue, and shader library.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    library: Library,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    /// Loads the pre-compiled metallib from OUT_DIR.
    pub fn new() -> MetalResult<Self> {
        let device = Device::system_default().ok_or(MetalError::DeviceNotFound)?;
        
        eprintln!("[Metal] Device: {:?}", device.name());
        eprintln!("[Metal] Recommended max working set: {} bytes ({:.1} GB)",
            device.recommended_max_working_set_size(),
            device.recommended_max_working_set_size() as f64 / 1e9);
        
        let queue = device.new_command_queue();

        // The metallib is EMBEDDED at compile time via include_bytes! — env!("OUT_DIR")
        // is the build-time OUT_DIR (the build script generates solver.metallib there
        // before this crate compiles). This makes MetalContext work from ANY downstream
        // crate (play-harness, bot-server): runtime OUT_DIR is unset there, so the old
        // file-load path silently failed and the GPU search fell back to CPU.
        // Fallback: an on-disk OUT_DIR/solver.metallib (e.g. if the embedded bytes are
        // ever stale), preserved for solver-core's own dev loop.
        const METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/solver.metallib"));
        let library = match device.new_library_with_data(METALLIB_BYTES) {
            Ok(lib) => {
                eprintln!("[Metal] Loaded embedded shader library ({} bytes)", METALLIB_BYTES.len());
                lib
            }
            Err(embed_err) => {
                let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
                let metallib_path = PathBuf::from(out_dir).join("solver.metallib");
                if metallib_path.exists() {
                    eprintln!("[Metal] Embedded load failed ({embed_err}); falling back to {}", metallib_path.display());
                    device.new_library_with_file(&metallib_path)
                        .map_err(|e| MetalError::LibraryCompilation(
                            format!("Failed to load {}: {}", metallib_path.display(), e)))?
                } else {
                    return Err(MetalError::LibraryCompilation(
                        format!("embedded metallib load failed: {embed_err}")));
                }
            }
        };
        
        Ok(MetalContext {
            device,
            queue,
            library,
        })
    }
    
    /// Get a reference to the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get a reference to the command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
    
    /// Get a kernel function by name.
    pub fn get_function(&self, name: &str) -> MetalResult<Function> {
        self.library.get_function(name, None)
            .map_err(|_| MetalError::FunctionNotFound(name.to_string()))
    }
    
    /// Create a compute pipeline state for a kernel function.
    pub fn create_pipeline(&self, function_name: &str) -> MetalResult<ComputePipelineState> {
        let function = self.get_function(function_name)?;
        self.device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreation(
                format!("{}: {}", function_name, e)
            ))
    }

    /// Create a pipeline SPECIALIZED via function constants (G6 lever
    /// 1): the compiler constant-folds the values — same ops, same
    /// order, bit-exact with the generic pipeline by construction.
    pub fn create_pipeline_with_constants(
        &self,
        function_name: &str,
        constants: &[(u64, u32)], // (function_constant index, value)
    ) -> MetalResult<ComputePipelineState> {
        let fcv = metal::FunctionConstantValues::new();
        for &(idx, v) in constants {
            fcv.set_constant_value_at_index(
                &v as *const u32 as *const std::ffi::c_void,
                metal::MTLDataType::UInt,
                idx,
            );
        }
        let function = self.library.get_function(function_name, Some(fcv))
            .map_err(|e| MetalError::FunctionNotFound(
                format!("{} (specialized): {}", function_name, e)
            ))?;
        self.device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError::PipelineCreation(
                format!("{} (specialized): {}", function_name, e)
            ))
    }
    
    /// Create a new command buffer.
    pub fn new_command_buffer(&self) -> &metal::CommandBufferRef {
        self.queue.new_command_buffer()
    }
    
    /// Allocate a zero-initialized buffer.
    pub fn alloc_zeros<T: Copy>(&self, len: usize) -> MetalBuffer<T> {
        MetalBuffer::zeros(&self.device, len)
    }
    
    /// Allocate a buffer and upload data.
    pub fn upload<T: Copy>(&self, data: &[T]) -> MetalBuffer<T> {
        MetalBuffer::from_slice(&self.device, data)
    }
    
    /// Compute thread group size and count for a 1D dispatch of `n` elements.
    pub fn dispatch_1d(&self, n: usize, max_threads_per_group: usize) -> (MTLSize, MTLSize) {
        let threads_per_group = max_threads_per_group.min(n).max(1);
        let thread_groups = (n + threads_per_group - 1) / threads_per_group;
        let group_size = MTLSize { width: threads_per_group as u64, height: 1, depth: 1 };
        let grid_size = MTLSize { width: thread_groups as u64, height: 1, depth: 1 };
        (grid_size, group_size)
    }
    
    /// Compute thread group size and count for a 2D dispatch.
    pub fn dispatch_2d(
        &self,
        width: usize, height: usize,
        max_threads_per_group: usize,
    ) -> (MTLSize, MTLSize) {
        // Use a 2D threadgroup that fits within max_threads_per_group
        let sqrt = (max_threads_per_group as f64).sqrt() as usize;
        let tg_width = sqrt.min(width).max(1);
        let tg_height = sqrt.min(height).max(1);
        // Make sure total doesn't exceed max
        let tg_total = tg_width * tg_height;
        let (tg_width, tg_height) = if tg_total > max_threads_per_group {
            let scale = (max_threads_per_group as f64 / tg_total as f64).sqrt();
            let tw = ((tg_width as f64 * scale) as usize).max(1);
            let th = ((tg_height as f64 * scale) as usize).max(1);
            (tw, th)
        } else {
            (tg_width, tg_height)
        };
        
        let groups_x = (width + tg_width - 1) / tg_width;
        let groups_y = (height + tg_height - 1) / tg_height;
        
        let grid_size = MTLSize { width: groups_x as u64, height: groups_y as u64, depth: 1 };
        let group_size = MTLSize { width: tg_width as u64, height: tg_height as u64, depth: 1 };
        (grid_size, group_size)
    }
}
