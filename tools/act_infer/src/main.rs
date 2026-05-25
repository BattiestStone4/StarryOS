use std::env;
use std::fs;
use std::io::{Read, Write};
use tract_onnx::prelude::*;

/// Load a tensor from the custom binary format:
///   [4B ndim (u32 LE)] [4B per dim (u32 LE)] [raw f32 data]
fn load_tensor(path: &str) -> TractResult<Tensor> {
    let mut buf = Vec::new();
    let mut f = fs::File::open(path)?;
    f.read_to_end(&mut buf)?;

    if buf.len() < 4 {
        anyhow::bail!("file too short: {}", path);
    }

    let ndim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if buf.len() < 4 + ndim * 4 {
        anyhow::bail!("file too short for shape: {}", path);
    }

    let mut shape: Vec<usize> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let off = 4 + i * 4;
        let dim = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        shape.push(dim);
    }

    let data_start = 4 + ndim * 4;
    let expected_len = shape.iter().product::<usize>() * 4;
    if buf.len() - data_start < expected_len {
        anyhow::bail!(
            "data too short: expected {} bytes, got {}",
            expected_len,
            buf.len() - data_start
        );
    }

    let float_count = shape.iter().product::<usize>();
    let mut data: Vec<f32> = Vec::with_capacity(float_count);
    for i in 0..float_count {
        let off = data_start + i * 4;
        let val = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        data.push(val);
    }

    let tensor = tract_ndarray::ArrayD::from_shape_vec(shape, data)?;
    Ok(tensor.into())
}

/// Save a tensor in the same custom binary format.
fn save_tensor(tensor: &Tensor, path: &str) -> TractResult<()> {
    let arr = tensor.to_array_view::<f32>()?;
    let shape = arr.shape();
    let mut buf: Vec<u8> = Vec::new();

    // ndim
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // shape
    for &dim in shape {
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    // data
    for &val in arr.as_slice().unwrap() {
        buf.extend_from_slice(&val.to_le_bytes());
    }

    let mut f = fs::File::create(path)?;
    f.write_all(&buf)?;
    Ok(())
}

fn main() -> TractResult<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <model.onnx> <input_dir> <output_path>", args[0]);
        eprintln!("  Reads <input_dir>/image.bin and <input_dir>/state.bin");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let input_dir = &args[2];
    let output_path = &args[3];

    let t_total = std::time::Instant::now();

    println!("Loading model from: {}", model_path);
    let t0 = std::time::Instant::now();
    let model = tract_onnx::onnx()
        .model_for_path(model_path)?
        .into_optimized()?
        .into_runnable()?;
    println!("Model loaded in {:.3}s", t0.elapsed().as_secs_f64());

    let t1 = std::time::Instant::now();
    let image_tensor = load_tensor(&format!("{}/image.bin", input_dir))?;
    let state_tensor = load_tensor(&format!("{}/state.bin", input_dir))?;
    println!("Input loaded in {:.3}s (image {:?}, state {:?})",
        t1.elapsed().as_secs_f64(), image_tensor.shape(), state_tensor.shape());

    println!("Running inference...");
    let t2 = std::time::Instant::now();
    let result = model.run(tvec!(image_tensor.into(), state_tensor.into()))?;
    println!("Inference done in {:.3}s", t2.elapsed().as_secs_f64());

    let output = &result[0];
    let arr = output.to_array_view::<f32>()?;
    println!("Output shape: {:?}", arr.shape());

    // Print first 7 values of output[0, 0, :]
    if arr.ndim() == 3 {
        let first_vals: Vec<f32> = arr.slice(tract_ndarray::s![0, 0, ..7]).to_vec();
        println!("First 7 values of output[0,0]: {:?}", first_vals);
    }

    // Compute stats
    let slice = arr.as_slice().unwrap();
    let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean = slice.iter().sum::<f32>() / slice.len() as f32;
    println!("Output min:  {:.6}", min);
    println!("Output max:  {:.6}", max);
    println!("Output mean: {:.6}", mean);

    // Save output
    println!("Saving output to: {}", output_path);
    save_tensor(output, output_path)?;
    println!("Total time: {:.3}s", t_total.elapsed().as_secs_f64());

    Ok(())
}
