use std::path::PathBuf;
use std::sync::Arc;

use metal::{
    MTLResourceOptions, MTLSize,
};

/// Metal buffer wrapper with typed access.
/// On Apple Silicon with unified memory, StorageModeShared gives
/// zero-copy access from both CPU and GPU.
pub struct MetalBuffer<T: Copy> {
    buffer: metal::Buffer,
    len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Copy> MetalBuffer<T> {
    /// Allocate a new buffer filled with zeros.
    pub fn zeros(device: &metal::Device, len: usize) -> Self {
        let size = len * std::mem::size_of::<T>();
        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        // Zero the buffer
        let ptr = buffer.contents() as *mut T;
        unsafe {
            std::ptr::write_bytes(ptr, 0, len);
        }
        MetalBuffer {
            buffer,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Allocate and upload data from a Vec.
    pub fn from_slice(device: &metal::Device, data: &[T]) -> Self {
        let len = data.len();
        let size = len * std::mem::size_of::<T>();
        let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        let ptr = buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
        }
        MetalBuffer {
            buffer,
            len,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the raw Metal buffer reference.
    pub fn as_ref(&self) -> &metal::BufferRef {
        &*self.buffer
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Download data to a Vec.
    pub fn to_vec(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len).to_vec() }
    }

    /// Get a mutable pointer to the buffer contents (CPU side).
    /// On unified memory, this is the same memory the GPU uses.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.buffer.contents() as *mut T;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len) }
    }

    /// Get a read-only slice of the buffer contents.
    pub fn as_slice(&self) -> &[T] {
        let ptr = self.buffer.contents() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Byte size of the buffer.
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

// Metal doesn't need Send/Sync guards for unified memory buffers
// because StorageModeShared is coherent by default.
unsafe impl<T: Copy + Send> Send for MetalBuffer<T> {}
unsafe impl<T: Copy + Sync> Sync for MetalBuffer<T> {}
