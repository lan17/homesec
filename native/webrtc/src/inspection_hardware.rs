//! Inspect linked components and validate explicitly selected decode backends.
//!
//! No installed executable contributes evidence. Device creation is camera-free;
//! only decoded hardware frames establish that a camera worked on that backend.

use std::{
    ffi::{CStr, CString},
    ptr,
};

use ffmpeg::{codec, ffi, frame::Video};
use ffmpeg_next as ffmpeg;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct DeviceCapability {
    pub backend: String,
    pub compiled: bool,
    pub available: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Capabilities {
    pub library_version: String,
    pub build_id: String,
    pub decoders: Vec<String>,
    pub encoders: Vec<String>,
    pub parsers: Vec<String>,
    pub muxers: Vec<String>,
    pub filters: Vec<String>,
    pub hwaccels: Vec<String>,
    pub device: Option<DeviceCapability>,
}

pub fn capabilities(backend: Option<&str>, device: Option<&str>) -> Capabilities {
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Quiet);
    let mut decoders = Vec::new();
    let mut encoders = Vec::new();
    let mut parsers = Vec::new();
    let mut muxers = Vec::new();
    let mut filters = Vec::new();
    let mut hwaccels = Vec::new();
    // The iterators return immutable process-lifetime library registrations.
    unsafe {
        let mut opaque = ptr::null_mut();
        loop {
            let codec = ffi::av_codec_iterate(&mut opaque);
            if codec.is_null() {
                break;
            }
            if ffi::av_codec_is_decoder(codec) != 0 {
                decoders.push(component_name((*codec).name));
            }
            if ffi::av_codec_is_encoder(codec) != 0 {
                encoders.push(component_name((*codec).name));
            }
        }
        let mut descriptor = ffi::avcodec_descriptor_next(ptr::null());
        while !descriptor.is_null() {
            let parser = ffi::av_parser_init((*descriptor).id as i32);
            if !parser.is_null() {
                parsers.push(component_name((*descriptor).name));
                ffi::av_parser_close(parser);
            }
            descriptor = ffi::avcodec_descriptor_next(descriptor);
        }
        opaque = ptr::null_mut();
        loop {
            let muxer = ffi::av_muxer_iterate(&mut opaque);
            if muxer.is_null() {
                break;
            }
            muxers.extend(component_name((*muxer).name).split(',').map(str::to_owned));
        }
        opaque = ptr::null_mut();
        loop {
            let filter = ffi::av_filter_iterate(&mut opaque);
            if filter.is_null() {
                break;
            }
            filters.push(component_name((*filter).name));
        }
        let mut previous = ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
        loop {
            previous = ffi::av_hwdevice_iterate_types(previous);
            if previous == ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                break;
            }
            hwaccels.push(component_name(ffi::av_hwdevice_get_type_name(previous)));
        }
    }
    let device = backend.map(|backend| {
        let compiled = hwaccels.iter().any(|name| name == backend);
        let error = if compiled {
            create_device(backend, device).err()
        } else {
            Some("local_capability_unavailable")
        };
        DeviceCapability {
            backend: backend.to_owned(),
            compiled,
            available: error.is_none(),
            reason: error.map(str::to_owned),
        }
    });
    for components in [
        &mut decoders,
        &mut encoders,
        &mut parsers,
        &mut muxers,
        &mut filters,
        &mut hwaccels,
    ] {
        components.sort_unstable();
        components.dedup();
    }
    Capabilities {
        library_version: unsafe { component_name(ffi::av_version_info()) },
        build_id: env!("HOMESEC_NATIVE_BUILD_ID").to_owned(),
        decoders,
        encoders,
        parsers,
        muxers,
        filters,
        hwaccels,
        device,
    }
}

unsafe fn component_name(name: *const std::ffi::c_char) -> String {
    // Names are compiled identifiers; never include libav's URL-bearing text.
    if name.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

struct HardwareDevice(*mut ffi::AVBufferRef);

impl Drop for HardwareDevice {
    fn drop(&mut self) {
        unsafe { ffi::av_buffer_unref(&mut self.0) };
    }
}

fn backend_type(backend: &str) -> Result<ffi::AVHWDeviceType, &'static str> {
    match backend {
        "vaapi" => Ok(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI),
        "cuda" => Ok(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA),
        "qsv" => Ok(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV),
        "videotoolbox" => Ok(ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX),
        _ => Err("invalid_hardware_backend"),
    }
}

fn backend_format(backend: ffi::AVHWDeviceType) -> ffi::AVPixelFormat {
    match backend {
        ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI => ffi::AVPixelFormat::AV_PIX_FMT_VAAPI,
        ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA => ffi::AVPixelFormat::AV_PIX_FMT_CUDA,
        ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV => ffi::AVPixelFormat::AV_PIX_FMT_QSV,
        ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX => {
            ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX
        }
        _ => ffi::AVPixelFormat::AV_PIX_FMT_NONE,
    }
}

fn create_device(backend: &str, device: Option<&str>) -> Result<HardwareDevice, &'static str> {
    let kind = backend_type(backend)?;
    let name = CString::new(backend).map_err(|_| "invalid_hardware_backend")?;
    if unsafe { ffi::av_hwdevice_find_type_by_name(name.as_ptr()) } != kind {
        return Err("local_capability_unavailable");
    }
    if device.is_some_and(|device| device.len() > 4096 || device.is_empty()) {
        return Err("invalid_hardware_device");
    }
    let device = device
        .map(CString::new)
        .transpose()
        .map_err(|_| "invalid_hardware_device")?;
    let mut options = ptr::null_mut();
    let mut context = ptr::null_mut();
    // QSV's device argument selects an implementation; a DRM path is its child
    // device option. CUDA and VAAPI accept their native device argument directly.
    let argument = if backend == "qsv" {
        if let Some(device) = &device {
            unsafe {
                ffi::av_dict_set(&mut options, c"child_device".as_ptr(), device.as_ptr(), 0);
            }
        }
        ptr::null()
    } else {
        device.as_ref().map_or(ptr::null(), |value| value.as_ptr())
    };
    let result = unsafe { ffi::av_hwdevice_ctx_create(&mut context, kind, argument, options, 0) };
    unsafe {
        ffi::av_dict_free(&mut options);
    }
    let context = HardwareDevice(context);
    if result < 0 {
        Err("local_hardware_unavailable")
    } else {
        Ok(context)
    }
}

/// Require a hardware pixel format. libav's default format negotiation permits
/// software fallback, which would falsely certify the requested hardware path.
unsafe extern "C" fn hardware_format(
    context: *mut ffi::AVCodecContext,
    offered: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    unsafe {
        if context.is_null() || offered.is_null() || (*context).hw_device_ctx.is_null() {
            return ffi::AVPixelFormat::AV_PIX_FMT_NONE;
        }
        let device = (*(*context).hw_device_ctx)
            .data
            .cast::<ffi::AVHWDeviceContext>();
        let expected = backend_format((*device).type_);
        // Codec format lists are terminated, with only a small finite set of
        // formats. Retain a defensive bound before following library pointers.
        for offset in 0..64 {
            let format = *offered.add(offset);
            if format == ffi::AVPixelFormat::AV_PIX_FMT_NONE {
                break;
            }
            if format == expected {
                return expected;
            }
        }
        ffi::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

pub fn open_video_decoder(
    parameters: &codec::Parameters,
    backend: &str,
    device: Option<&str>,
) -> Result<codec::decoder::Video, &'static str> {
    let decoder = if backend == "qsv" {
        codec::decoder::find_by_name(&format!("{}_qsv", parameters.id().name()))
    } else {
        codec::decoder::find(parameters.id())
    }
    .ok_or("local_capability_unavailable")?;
    let mut context = codec::Context::from_parameters(parameters.clone())
        .map_err(|_| "inspection_decode_failed")?
        .decoder();
    context.set_threading(codec::threading::Config::count(1));
    if backend != "software" {
        let kind = backend_type(backend)?;
        let mut supported = false;
        for index in 0..64 {
            let config = unsafe { ffi::avcodec_get_hw_config(decoder.as_ptr(), index) };
            if config.is_null() {
                break;
            }
            if unsafe {
                (*config).device_type == kind
                    && (*config).methods & ffi::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32 != 0
            } {
                supported = true;
                break;
            }
        }
        if !supported {
            return Err("local_capability_unavailable");
        }
        let device = create_device(backend, device)?;
        unsafe {
            (*context.as_mut_ptr()).hw_device_ctx = ffi::av_buffer_ref(device.0);
            if (*context.as_mut_ptr()).hw_device_ctx.is_null() {
                return Err("local_hardware_unavailable");
            }
            (*context.as_mut_ptr()).get_format = Some(hardware_format);
        }
    }
    context
        .open_as(decoder)
        .and_then(|decoder| decoder.video())
        .map_err(|_| "inspection_decode_failed")
}

pub fn decoded_frame_backend(frame: &Video, backend: &str) -> bool {
    unsafe {
        if backend == "software" {
            return (*frame.as_ptr()).hw_frames_ctx.is_null();
        }
        let Ok(kind) = backend_type(backend) else {
            return false;
        };
        !(*frame.as_ptr()).hw_frames_ctx.is_null()
            && ffi::AVPixelFormat::from(frame.format()) == backend_format(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_capabilities_preserve_software_validation_components() {
        // Given: The linked pinned library, with no camera or host CLI.
        // When: Inspecting its compiled components.
        let result = capabilities(None, None);
        // Then: Software camera/audio validation is complete and no device was opened.
        assert!(result.library_version.starts_with("8.1.3"));
        assert!(result.build_id.starts_with("ffmpeg-8.1.3-"));
        for name in [
            "h264",
            "hevc",
            "aac",
            "aac_latm",
            "pcm_alaw",
            "pcm_mulaw",
            "opus",
            "ac3",
            "eac3",
            "alac",
            "mp2",
            "mp3",
        ] {
            assert!(
                result.decoders.iter().any(|decoder| decoder == name),
                "missing {name}"
            );
        }
        assert!(result.encoders.iter().any(|encoder| encoder == "aac"));
        assert!(result.parsers.iter().any(|parser| parser == "hevc"));
        assert!(result.filters.iter().any(|filter| filter == "aresample"));
        assert!(result.muxers.iter().any(|muxer| muxer == "mp4"));
        assert!(result.device.is_none());
    }

    #[test]
    fn invalid_or_missing_hardware_never_reports_device_success() {
        // Given: A backend name absent from the linked hardware registration.
        // When: Asking for camera-free device evidence.
        let result = capabilities(Some("unavailable_backend"), None);
        // Then: It is a local capability refusal, independent of camera connectivity.
        let device = result.device.unwrap();
        assert!(!device.compiled);
        assert!(!device.available);
        assert_eq!(
            device.reason.as_deref(),
            Some("local_capability_unavailable")
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_build_has_adapters_and_rejects_a_missing_render_device() {
        // Given: Linux adapter code is compiled without requiring a GPU.
        let build = capabilities(None, None);
        for backend in ["vaapi", "cuda", "qsv"] {
            assert!(build.hwaccels.iter().any(|name| name == backend));
        }
        for encoder in ["h264_vaapi", "h264_nvenc", "h264_qsv"] {
            assert!(build.encoders.iter().any(|name| name == encoder));
        }
        // When: VAAPI is asked to open a render device that does not exist.
        let result = capabilities(Some("vaapi"), Some("/dev/homesec-missing-render-device"));
        // Then: Build support stays distinct from unavailable host resources.
        let device = result.device.unwrap();
        assert!(device.compiled);
        assert!(!device.available);
        assert_eq!(device.reason.as_deref(), Some("local_hardware_unavailable"));
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_build_includes_videotoolbox_without_claiming_camera_support() {
        // Given: The pinned library was linked against the platform framework.
        // When: Camera-free compiled support is enumerated.
        let result = capabilities(None, None);
        // Then: Both adapter and encoder registrations exist, without decode evidence.
        assert!(result.hwaccels.iter().any(|name| name == "videotoolbox"));
        assert!(
            result
                .encoders
                .iter()
                .any(|name| name == "h264_videotoolbox")
        );
        assert!(result.device.is_none());
    }

    #[test]
    fn software_frames_cannot_be_hardware_decode_evidence() {
        // Given: An ordinary software frame.
        let frame = Video::new(ffmpeg::format::Pixel::YUV420P, 16, 16);
        // When: Its actual frame backend is checked.
        // Then: Only the explicit software backend accepts it as decode evidence.
        assert!(decoded_frame_backend(&frame, "software"));
        for backend in ["vaapi", "cuda", "qsv", "videotoolbox"] {
            assert!(!decoded_frame_backend(&frame, backend));
        }
    }
}
