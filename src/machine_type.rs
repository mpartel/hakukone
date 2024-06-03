use anyhow::Result;
use libc::utsname;
use std::{ffi::CStr, io};

pub fn machine_type() -> Result<String> {
    let mut data: utsname = unsafe { std::mem::zeroed() };
    if unsafe { libc::uname(&mut data) } == 0 {
        Ok(unsafe {
            CStr::from_ptr(data.machine.as_ptr())
                .to_string_lossy()
                .into()
        })
    } else {
        Err(io::Error::last_os_error().into())
    }
}
