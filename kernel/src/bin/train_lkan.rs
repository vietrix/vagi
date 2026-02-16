use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss};
use rand::Rng;
use vagi_kernel::model::gpt_kan::{LKanGPT, LKanGPTConfig};
use vagi_kernel::model::lkan::LiquidKanConfig;

const INPUT_PATH: &str = "data/input.txt";
const OUTPUT_PATH: &str = "models/lkan-genesis.safetensors";

const BATCH_SIZE: usize = 32;
const SEQ_LEN: usize = 64;
const TRAIN_STEPS: usize = 1_000;
const LOG_EVERY: usize = 100;
const GENERATE_TOKENS: usize = 50;

#[derive(Debug, Clone)]
struct CharTokenizer {
    stoi: HashMap<char, u32>,
    itos: Vec<char>,
}

impl CharTokenizer {
    fn from_text(text: &str) -> Result<Self> {
        let charset: BTreeSet<char> = text.chars().collect();
        if charset.is_empty() {
            bail!("training text is empty, cannot build character tokenizer");
        }

        let itos: Vec<char> = charset.into_iter().collect();
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(idx, ch)| (*ch, idx as u32))
            .collect();
        Ok(Self { stoi, itos })
    }

    fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(text.chars().count());
        for ch in text.chars() {
            let id =
                self.stoi.get(&ch).copied().with_context(|| {
                    format!("character `{ch}` not found in tokenizer vocabulary")
                })?;
            ids.push(id);
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            let ch = self
                .itos
                .get(id as usize)
                .copied()
                .with_context(|| format!("token id {id} out of vocabulary range"))?;
            out.push(ch);
        }
        Ok(out)
    }
}

fn sample_batch(
    tokens: &[u32],
    batch_size: usize,
    seq_len: usize,
    rng: &mut impl Rng,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    if tokens.len() <= seq_len + 1 {
        bail!(
            "tokenized corpus too small: len={} but need > {}",
            tokens.len(),
            seq_len + 1
        );
    }

    let max_start = tokens.len() - seq_len - 1;
    let mut x_buf = Vec::with_capacity(batch_size * seq_len);
    let mut y_buf = Vec::with_capacity(batch_size * seq_len);

    for _ in 0..batch_size {
        let start = rng.random_range(0..=max_start);
        let x_slice = &tokens[start..start + seq_len];
        let y_slice = &tokens[start + 1..start + seq_len + 1];
        x_buf.extend_from_slice(x_slice);
        y_buf.extend_from_slice(y_slice);
    }

    let x = Tensor::from_slice(&x_buf, (batch_size, seq_len), device)?;
    let y = Tensor::from_slice(&y_buf, (batch_size, seq_len), device)?;
    Ok((x, y))
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_val {
            best_idx = idx;
            best_val = value;
        }
    }
    best_idx
}

fn generate_text(
    model: &LKanGPT,
    tokenizer: &CharTokenizer,
    seed: &str,
    max_new_tokens: usize,
    context_len: usize,
    device: &Device,
) -> Result<String> {
    let mut ids = tokenizer.encode(seed)?;
    if ids.is_empty() {
        bail!("seed text for generation must not be empty");
    }

    for _ in 0..max_new_tokens {
        let start = ids.len().saturating_sub(context_len);
        let context_ids = &ids[start..];
        let input = Tensor::from_slice(context_ids, (1, context_ids.len()), device)?;
        let logits = model.forward_logits(&input)?;
        let last_logits = logits
            .narrow(1, context_ids.len() - 1, 1)?
            .reshape((tokenizer.vocab_size(),))?;
        let probs = last_logits.to_vec1::<f32>()?;
        let next_id = argmax(&probs) as u32;
        ids.push(next_id);
    }

    tokenizer.decode(&ids)
}

fn main() -> Result<()> {
    let input_path = Path::new(INPUT_PATH);
    if !input_path.exists() {
        bail!(
            "missing `{}`. Download TinyShakespeare and save it at this path.\nURL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            input_path.display()
        );
    }

    let text = fs::read_to_string(input_path)
        .with_context(|| format!("failed to read {}", input_path.display()))?;
    if text.trim().is_empty() {
        bail!("input corpus is empty after trimming whitespace");
    }

    let tokenizer = CharTokenizer::from_text(&text)?;
    let token_ids = tokenizer.encode(&text)?;
    if token_ids.len() <= SEQ_LEN + 1 {
        bail!(
            "corpus too short for seq_len={}: need > {} tokens but got {}",
            SEQ_LEN,
            SEQ_LEN + 1,
            token_ids.len()
        );
    }

    let device = Device::Cpu;
    let vocab_size = tokenizer.vocab_size();
    println!(
        "dataset loaded: chars={} vocab={} batch_size={} seq_len={}",
        text.chars().count(),
        vocab_size,
        BATCH_SIZE,
        SEQ_LEN
    );

    let config = LKanGPTConfig {
        vocab_size,
        hidden_dim: 128,
        num_layers: 4,
        num_heads: 4,
        kan_config: LiquidKanConfig {
            in_dim: 128,
            hidden_dim: 128,
            out_dim: 128,
            cheb_order: 5,
            dt: 0.05,
            tau_min: 1e-3,
            x_scale: 2.0,
        },
    };

    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let model =
        LKanGPT::new(vb.pp("lkan_gpt"), config.clone()).context("failed to initialize LKanGPT")?;

    let mut optimizer = AdamW::new(
        var_map.all_vars(),
        ParamsAdamW {
            lr: 3e-4,
            weight_decay: 1e-2,
            ..Default::default()
        },
    )?;

    let mut rng = rand::rng();
    for step in 1..=TRAIN_STEPS {
        let (x, y) = sample_batch(&token_ids, BATCH_SIZE, SEQ_LEN, &mut rng, &device)?;
        let logits = model.forward_logits(&x)?;
        let logits = logits.reshape((BATCH_SIZE * SEQ_LEN, vocab_size))?;
        let targets = y.reshape((BATCH_SIZE * SEQ_LEN,))?;
        let loss = loss::cross_entropy(&logits, &targets)?;
        let loss_value = loss.to_scalar::<f32>()?;

        optimizer.backward_step(&loss)?;

        if step % LOG_EVERY == 0 {
            println!("step {:4} | loss {:.6}", step, loss_value);
        }
    }

    let output_path = Path::new(OUTPUT_PATH);
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    var_map
        .save(output_path)
        .with_context(|| format!("failed to save checkpoint to {}", output_path.display()))?;
    println!("saved checkpoint to {}", output_path.display());

    let seed_len = text.chars().count().min(SEQ_LEN.max(1));
    let seed: String = text.chars().take(seed_len).collect();
    let generated = generate_text(&model, &tokenizer, &seed, GENERATE_TOKENS, SEQ_LEN, &device)?;
    println!(
        "---- generated text ({GENERATE_TOKENS} new chars, greedy) ----\n{}",
        generated
    );

    Ok(())
}
