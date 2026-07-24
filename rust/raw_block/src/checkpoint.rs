// SPDX-License-Identifier: Apache-2.0

//! Raw-block checkpoint JSON serialization.
//!
//! The Python raw-block index remains the source of truth. This module only
//! converts a detached, shallow copy of that index directly into the compact
//! UTF-8 JSON bytes stored in the checkpoint containers. Building the payload
//! here avoids materializing an additional per-entry Python dictionary and
//! Python string before producing the final bytes.

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyIterator, PyList, PyTuple};
use std::borrow::Cow;
use std::io::Write;

fn serialization_error(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(format!(
        "raw-block checkpoint JSON serialization failed: {}",
        message.into()
    ))
}

fn write_integer<T>(payload: &mut Vec<u8>, value: T) -> PyResult<()>
where
    T: std::fmt::Display,
{
    write!(payload, "{value}").map_err(|error| serialization_error(error.to_string()))
}

fn write_unicode_escape(payload: &mut Vec<u8>, value: u16) {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    payload.extend_from_slice(b"\\u");
    payload.push(HEX[((value >> 12) & 0x0f) as usize]);
    payload.push(HEX[((value >> 8) & 0x0f) as usize]);
    payload.push(HEX[((value >> 4) & 0x0f) as usize]);
    payload.push(HEX[(value & 0x0f) as usize]);
}

/// Write a JSON string using the escaping produced by
/// `json.dumps(value, ensure_ascii=True)`.
fn write_json_string(payload: &mut Vec<u8>, value: &str) {
    payload.push(b'"');
    for character in value.chars() {
        match character {
            '"' => payload.extend_from_slice(b"\\\""),
            '\\' => payload.extend_from_slice(b"\\\\"),
            '\u{0008}' => payload.extend_from_slice(b"\\b"),
            '\u{000c}' => payload.extend_from_slice(b"\\f"),
            '\n' => payload.extend_from_slice(b"\\n"),
            '\r' => payload.extend_from_slice(b"\\r"),
            '\t' => payload.extend_from_slice(b"\\t"),
            '\u{0020}'..='\u{007e}' => payload.push(character as u8),
            character if (character as u32) <= 0xffff => {
                write_unicode_escape(payload, character as u16);
            }
            character => {
                let code_point = character as u32 - 0x1_0000;
                let high_surrogate = 0xd800 + (code_point >> 10);
                let low_surrogate = 0xdc00 + (code_point & 0x03ff);
                write_unicode_escape(payload, high_surrogate as u16);
                write_unicode_escape(payload, low_surrogate as u16);
            }
        }
    }
    payload.push(b'"');
}

fn normalize_dtype_name(value: &str) -> Cow<'_, str> {
    match value {
        "torch.half" | "torch.float16" => Cow::Borrowed("half"),
        "torch.bfloat16" => Cow::Borrowed("bfloat16"),
        "torch.float" | "torch.float32" => Cow::Borrowed("float"),
        "torch.double" | "torch.float64" => Cow::Borrowed("double"),
        "torch.int8" => Cow::Borrowed("int8"),
        "torch.uint8" => Cow::Borrowed("uint8"),
        "torch.int16" => Cow::Borrowed("int16"),
        "torch.int32" => Cow::Borrowed("int32"),
        "torch.int64" => Cow::Borrowed("int64"),
        "torch.bool" => Cow::Borrowed("bool"),
        "torch.float8_e4m3fn" => Cow::Borrowed("fp8_e4m3fn"),
        "torch.float8_e4m3fnuz" => Cow::Borrowed("fp8_e4m3fnuz"),
        "torch.float8_e5m2" => Cow::Borrowed("fp8_e5m2"),
        "torch.float8_e5m2fnuz" => Cow::Borrowed("fp8_e5m2fnuz"),
        _ => Cow::Borrowed(value),
    }
}

fn write_dtype(payload: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() {
        payload.extend_from_slice(b"null");
        return Ok(());
    }

    let string_value = value.str()?;
    let dtype_name = normalize_dtype_name(string_value.to_str()?);
    write_json_string(payload, dtype_name.as_ref());
    Ok(())
}

fn write_format(payload: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() {
        payload.extend_from_slice(b"null");
        return Ok(());
    }

    match value.getattr("name") {
        Ok(name) => {
            let name: &str = name.extract()?;
            write_json_string(payload, name);
        }
        Err(error) if error.is_instance_of::<PyAttributeError>(value.py()) => {
            let string_value = value.str()?;
            write_json_string(payload, string_value.to_str()?);
        }
        Err(error) => return Err(error),
    }
    Ok(())
}

fn write_shape(payload: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() {
        payload.extend_from_slice(b"null");
        return Ok(());
    }

    payload.push(b'[');
    let mut first = true;
    for item in PyIterator::from_object(value)? {
        if first {
            first = false;
        } else {
            payload.push(b',');
        }
        write_integer(payload, item?.extract::<i64>()?)?;
    }
    payload.push(b']');
    Ok(())
}

fn write_integer_tree(payload: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() {
        payload.extend_from_slice(b"null");
        return Ok(());
    }

    if value.is_instance_of::<PyBool>() {
        if value.extract::<bool>()? {
            payload.extend_from_slice(b"true");
        } else {
            payload.extend_from_slice(b"false");
        }
        return Ok(());
    }

    if let Ok(items) = value.downcast::<PyList>() {
        payload.push(b'[');
        for (index, item) in items.iter().enumerate() {
            if index > 0 {
                payload.push(b',');
            }
            write_integer_tree(payload, &item)?;
        }
        payload.push(b']');
        return Ok(());
    }

    if let Ok(items) = value.downcast::<PyTuple>() {
        payload.push(b'[');
        for (index, item) in items.iter().enumerate() {
            if index > 0 {
                payload.push(b',');
            }
            write_integer_tree(payload, &item)?;
        }
        payload.push(b']');
        return Ok(());
    }

    if let Ok(integer) = value.extract::<i64>() {
        return write_integer(payload, integer);
    }

    Err(PyTypeError::new_err(
        "cached_positions.tolist() must return integers or nested lists of integers",
    ))
}

fn write_cached_positions(payload: &mut Vec<u8>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() || !value.hasattr("tolist")? {
        payload.extend_from_slice(b"null");
        return Ok(());
    }

    let positions = value.call_method0("tolist")?;
    write_integer_tree(payload, &positions)
}

/// Serialize a raw-block metadata index directly into checkpoint JSON bytes.
///
/// `index` must be a detached shallow copy captured together with
/// `data_base_offset`, `next_slot`, and the dirty counter while the Python
/// `RawBlockCore` lock is held. Its values use the private Python `_Entry`
/// shape: `entry.offset` and `entry.meta.{size,shape,dtype,fmt,cached_positions}`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn serialize_raw_block_checkpoint_payload(
    py: Python<'_>,
    device_path: &str,
    capacity_bytes: u64,
    block_align: u64,
    header_bytes: u64,
    slot_bytes: u64,
    meta_total_bytes: u64,
    meta_magic: &str,
    meta_version: u64,
    data_base_offset: u64,
    next_slot: u64,
    index: &Bound<'_, PyDict>,
) -> PyResult<Py<PyBytes>> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"{\"version\":1,\"device_path\":");
    write_json_string(&mut payload, device_path);
    payload.extend_from_slice(b",\"capacity_bytes\":");
    write_integer(&mut payload, capacity_bytes)?;
    payload.extend_from_slice(b",\"block_align\":");
    write_integer(&mut payload, block_align)?;
    payload.extend_from_slice(b",\"header_bytes\":");
    write_integer(&mut payload, header_bytes)?;
    payload.extend_from_slice(b",\"slot_bytes\":");
    write_integer(&mut payload, slot_bytes)?;
    payload.extend_from_slice(b",\"meta_total_bytes\":");
    write_integer(&mut payload, meta_total_bytes)?;
    payload.extend_from_slice(b",\"meta_magic\":");
    write_json_string(&mut payload, meta_magic);
    payload.extend_from_slice(b",\"meta_version\":");
    write_integer(&mut payload, meta_version)?;
    payload.extend_from_slice(b",\"data_base_offset\":");
    write_integer(&mut payload, data_base_offset)?;
    payload.extend_from_slice(b",\"next_slot\":");
    write_integer(&mut payload, next_slot)?;
    payload.extend_from_slice(b",\"entries\":{");

    for (entry_index, (encoded_key, entry)) in index.iter().enumerate() {
        if entry_index > 0 {
            payload.push(b',');
        }

        write_json_string(&mut payload, encoded_key.extract::<&str>()?);
        payload.extend_from_slice(b":{\"offset\":");
        write_integer(&mut payload, entry.getattr("offset")?.extract::<u64>()?)?;

        let metadata = entry.getattr("meta")?;
        payload.extend_from_slice(b",\"size\":");
        write_integer(&mut payload, metadata.getattr("size")?.extract::<u64>()?)?;
        payload.extend_from_slice(b",\"shape\":");
        write_shape(&mut payload, &metadata.getattr("shape")?)?;
        payload.extend_from_slice(b",\"dtype\":");
        write_dtype(&mut payload, &metadata.getattr("dtype")?)?;
        payload.extend_from_slice(b",\"fmt\":");
        write_format(&mut payload, &metadata.getattr("fmt")?)?;
        payload.extend_from_slice(b",\"cached_positions\":");
        write_cached_positions(&mut payload, &metadata.getattr("cached_positions")?)?;
        payload.push(b'}');
    }

    payload.extend_from_slice(b"}}");
    Ok(PyBytes::new(py, &payload).unbind())
}

#[cfg(test)]
mod tests {
    use super::{normalize_dtype_name, write_json_string};

    #[test]
    fn test_write_json_string_matches_ensure_ascii() {
        let mut payload = Vec::new();
        write_json_string(
            &mut payload,
            "\0\u{0008}\t\n\u{000c}\r\"\\ /~\u{007f}\u{0080}é€😀",
        );

        assert_eq!(
            payload,
            br#""\u0000\b\t\n\f\r\"\\ /~\u007f\u0080\u00e9\u20ac\ud83d\ude00""#
        );
    }

    #[test]
    fn test_normalize_dtype_name_matches_python_aliases() {
        for (raw_name, checkpoint_name) in [
            ("torch.half", "half"),
            ("torch.float16", "half"),
            ("torch.bfloat16", "bfloat16"),
            ("torch.float", "float"),
            ("torch.float32", "float"),
            ("torch.double", "double"),
            ("torch.float64", "double"),
            ("torch.int8", "int8"),
            ("torch.uint8", "uint8"),
            ("torch.int16", "int16"),
            ("torch.int32", "int32"),
            ("torch.int64", "int64"),
            ("torch.bool", "bool"),
            ("torch.float8_e4m3fn", "fp8_e4m3fn"),
            ("torch.float8_e4m3fnuz", "fp8_e4m3fnuz"),
            ("torch.float8_e5m2", "fp8_e5m2"),
            ("torch.float8_e5m2fnuz", "fp8_e5m2fnuz"),
        ] {
            assert_eq!(normalize_dtype_name(raw_name), checkpoint_name);
        }
        assert_eq!(normalize_dtype_name("torch.complex64"), "torch.complex64");
    }
}
