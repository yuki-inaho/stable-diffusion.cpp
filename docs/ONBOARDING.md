# 環境セットアップ・オンボーディングガイド（stable-diffusion.cpp + MiniMax-H3 GGUF）

## 目次

1. プロジェクト概要
2. クリティカルな要求・制約
3. 現在のプロジェクト状態
4. 前提条件の確認
5. 環境セットアップ手順
   5.1 pixi によるビルド環境
   5.2 CUDA ビルド
6. モデルの取得・配置
7. 動作確認（FL2VA 生成）
8. トラブルシューティング
9. 次のステップ
10. 環境セットアップ完了チェックリスト
11. 更新履歴
12. このドキュメントについて

---

## 1. プロジェクト概要

### プロジェクト名

`stable-diffusion.cpp`（MiniMax-H3 GGUF 対応ビルド）

- **スクープ**: [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) の MiniMax-H3 対応コミット `ea7f0c8`（PR #1854 マージ）
- **フォーク**: `yuki-inaho/stable-diffusion.cpp`（本ガイドの検証・ビルド環境）
- **検証済みホスト**: Linux / RTX 5090（sm_120）/ Driver 580.173.02（CUDA 13.0）

### 最終目標

MiniMax-H3 の GGUF モデルを stable-diffusion.cpp でネイティブ実行し、**T2VA / I2VA / FL2VA（first+last frame） / Ref2VA** の動画＋音声生成ができる環境を、再現可能に保つこと。

### 主要コンポーネント

| コンポーネント | 内容 |
| :--- | :--- |
| 実行バイナリ | `sd-cli` / `sd-server`（CUDA Release ビルド） |
| ビルド環境 | pixi（CUDA Toolkit 13.0.3 / cmake 4.4.2 / ninja / gcc 14.4.0） |
| DiT (GGUF) | `minimax_h3_fl2va_pruned-Q4_K_M.gguf`（leejet 版・pruned AdaLN） |
| テキストエンコーダ (GGUF) | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`（leejet 版・vision tower 同梱） |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors`（Comfy-Org） |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors`（Comfy-Org） |

---

## 2. クリティカルな要求・制約

- **テキストエンコーダは MiniMax-H3 専用の Qwen3-VL-32B 変種**（50 言語レイヤーへ truncate し、最終 LM norm を除いて export）であること。vision tower（DeepStack merger 3 つ込み）が別ファイルなら `--llm_vision` で渡す。
- **DiT は pruned AdaLN（曲線テーブル）版と time-embedder 版の両方を重みから自動検出**するが、検出が失敗する（形状不一致）GGUF がある。
- **Abiray/MiniMax-H3-GGUF の Q5_K_M は非pruned ベース**のため、sd.cpp の期待形状と不一致になりエラー（後述）。**leejet/MiniMax-H3-GGUF を推奨**（公式検証対象）。
- **音声は joint AV 経路**。`--audio-vae` を渡すと 32 kHz ステレオ音声が復号される（省略は映像のみ）。
- **フレーム数は 17k+5 グリッドに合わせられ、最小 5。** 幅高さは 32 の倍数へ切り上げ。**FPS は 24 固定**（他の指定値は無視）。
- **メモリ**: DiT Q4(~11GB) + TE Q4(~18GB) + VAE 群で 32GB VRAM に同時常駐しない可能性 → `--offload-to-cpu` と条件の先出し計算を推奨。

---

## 3. 現在のプロジェクト状態

### 完了済み

- pixi ビルド環境追加（コミット `1a487e4`）
- GitHub Actions `release-builds.yml`（A100 CUDA sm_80 / R6000 ROCm gfx1100）（コミット `306780f`）
- CUDA 13.0.3 で `ea7f0c8`（H3 対応）をビルドし、sm_120 コンパイル・実行を確認
- RTX 5090 向けビルド済みバイナリを Release `v0.1.0-h3` に格納
- **FL2VA（first+last frame）生成を実機検証**（下記 §7、101.02s）

### 未着手・今後

- Ref2VA（参照画像・動画・音声）の実機検証は未着手
- A100 / R6000 向け CI の実実行・成果物確認は未実施
- Diffusers への H3 GGUF ローダー（LTX-2 の `from_single_file` 方式）は未実装（方針は stable-diffusion.cpp を実行エンジン化）

### 重要なファイル

| パス | 説明 |
| :--- | :--- |
| `pixi.toml` / `pixi.lock` | ビルド環境定義（CUDA 13.0.3） |
| `.github/workflows/release-builds.yml` | release 用 CI |
| `docs/minimax_h3.md` | 上流の H3 実行例 |
| `docs/build.md` | ビルド手順（本ガイドと重複） |
| `docs/ONBOARDING.md` | 本ガイド |

---

## 4. 前提条件の確認

```bash
nvidia-smi            # NVIDIA driver / CUDA バージョン
df -h ~/Desktop       # モデル 30GB 前後の空き
# システム側 CUDA が古くても pixi が nvcc を持ってくる（下記 §5）
```

`pixi` が無い場合は:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"
```

---

## 5. 環境セットアップ手順

### 5.1 pixi によるビルド環境

本リポジトリ（`yuki-inaho/stable-diffusion.cpp` or 上流）直下に:

```sh
export PATH="$HOME/.pixi/bin:$PATH"
pixi install   # pixi.toml に定義された環境（CUDA 13.0.3 等）を構築
pixi run nvcc --version    # Cuda compilation tools, release 13.0
pixi run cmake --version   # cmake 4.4.2
```

### 5.2 CUDA ビルド

```sh
export PATH="$HOME/.pixi/bin:$PATH"
pixi run build   # (= cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON && cmake --build build)
# 生成物: build/bin/sd-cli, build/bin/sd-server
```

注意: システムの `/usr/local/cuda` が古い（例: 11.8）場合でも pixi 内の nvcc が使われます。ggml-cuda は CMake >= 3.18 を要求するため、pixi の cmake（3.24+）を使用すること。

---

## 6. モデルの取得・配置

推奨: **leejet/MiniMax-H3-GGUF**（sd.cpp の公式検証対象）。

```bash
source ~/.bashrc        # HF_TOKEN

# DiT (pruned)
uv run hf download leejet/MiniMax-H3-GGUF \
  --include "minimax_h3_fl2va_pruned-Q4_K_M.gguf" \
  --local-dir ~/Desktop/models/h3_gguf/leejet

# テキストエンコーダ（vision tower 同梱）
uv run hf download leejet/MiniMax-H3-GGUF \
  --include "qwen3vl_32b_minimax_h3-Q4_K_M.gguf" \
  --local-dir ~/Desktop/models/h3_gguf/leejet

# VAE（safetensors、Comfy-Org のもの）
# video: minimax_h3_video_vae_fp16.safetensors
# audio: minimax_h3_audio_vae_fp32.safetensors
```

**非推奨**: `Abiray/MiniMax-H3-GGUF` の `MiniMax-H3-FL2VA-Q5_K_M.gguf` は非pruned ベース（Discussion #3 で確認）で、sd.cpp の形状検証に失敗する。

---

## 7. 動作確認（FL2VA 生成）

まず first/last frame 画像（ここでは 864×480）を用意し、`--init-img` + `--end-img` で実行します。

```sh
sd-cli/bin の場所: build/bin/sd-cli

build/bin/sd-cli -M vid_gen \
  --diffusion-model ~/Desktop/models/h3_gguf/leejet/minimax_h3_fl2va_pruned-Q4_K_M.gguf \
  --vae         ~/Desktop/models/h3_gguf/../minimax-h3_sandbox/models/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae   ~/Desktop/models/h3_gguf/../minimax-h3_sandbox/models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --llm         ~/Desktop/models/h3_gguf/leejet/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  -p "integrated_multimodal_description: [Shot 1] A bright sunny mountain landscape at noon. [Shot 2] At 00:01.500, the scene transitions to a warm orange-red sunset.\noverall_soundscape: gentle wind, birds fading to evening crickets.\nnon_diegetic_music: N/A" \
  -i first_frame.png --end-img last_frame.png \
  --cfg-scale 1.0 -W 864 -H 480 --diffusion-fa --offload-to-cpu --rng cpu \
  --fps 24 --video-frames 56 -o fl2v_test.mp4
```

### 実測結果（RTX 5090, 2026-08-05）

- `get_learned_condition`: 6.81s
- `generate_video 864x480x56` → **sampling 81.4s / 計 101.0s**
- **audio VAE decode 成功**（1.78s）
- 出力: 56 フレーム / 24fps / 864×480 / **32 kHz ステレオ PCM 音声 1 本**

フレーム遷移（平均色）: 先頭 (99,144,133) → 中間 (127,64,12) → 末尾 (134,67,12) と、昼→夕焼けへ遷移＝first/last frame interpolation が機能していることを確認。

---

## 8. トラブルシューティング

### 問題1: `model metadata validation failed`（DiT）

`model.diffusion_model.blocks.N.adaln_proj.linear.weight` の形状が `got [256,1016064]` / `expected [225792,96768]` のように不一致。

**原因**: Abiray 版 GGUF（MiniMax-H3-FL2VA-Q5_K_M.gguf）は非pruned ベース（`adaln_t_table` 無し）で、sd.cpp の pruned 期待形状と不一致。

**解決**: `leejet/MiniMax-H3-GGUF` の `minimax_h3_fl2va_pruned-Q4_K_M.gguf`（`adaln_t_table` を含む）を使う。確認は gguf パーサで `adaln_t_table` の有無をチェック。

### 問題2: Conditioner（text encoder）で形状不一致

`text_encoders.llm.visual.*` の形状一致しない（例: `visual.patch_embed.proj.weight got [16,16,6,1152] expected [16,16,2,3456]`）。

**原因**: Abiray 版テキストエンコーダ GGUF（`text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`）は 3-in-1 の変換で、sd.cpp が想定する MiniMax-H3 版 vision tower 形状と不一致。

**解決**: `leejet/MiniMax-H3-GGUF` の `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`（`visual.*` 351 テンソル同梱）を使う。`--llm_vision` は vision が分離されている場合のみ。

### 問題3: フレーム数・FPS が想定外

- フレーム数は `17k+5` グリッドに補正される（例: 56 → 57? 実測は 56）。
- FPS は 24 固定（`--fps` は無視される）。
- 幅高さは 32 の倍数に切り上げ。

### 問題4: VRAM 不足 / OOM

`--offload-to-cpu`（重みを RAM に置き、必要時に VRAM へ）+ `--diffusion-fa` を推奨。DiT Q4(~11G) + TE Q4(~18G) + VAE を同時に VRAM へ載せない。

### 問題5: 出力が `.avi` になる

`-o out.mp4` 指定でもコンテナが AVI になることがある（MJPEG/PCM の組み合わせ）。プレイヤー対応を承知。mux 先形式が必要なら `-o out.webm` 等を試す。

---

## 9. 次のステップ

1. Ref2VA（参照画像/動画/音声）の実機検証
2. A100 / R6000 向け CI の実実行確認（`workflow_dispatch` でタグ既存リリースへ添付）
3. Python ラッパー（`sd_vid_gen_params_t` 6 を C API で薄く包み、first/last 画像・sampler/scheduler・seed を制御）を作る
4. 必要なら条件 cache（prompt/first/last/hash をキーに）を実装

---

## 10. 環境セットアップ完了チェックリスト

- [ ] `pixi` 導入（`pixi --version`）
- [ ] `pixi install` 成功（nvcc 13.0.x / cmake 4.4.2）
- [ ] `pixi run build` 成功（SD_CUDA=ON、`build/bin/sd-cli` 生成）
- [ ] `sd-cli --version` が `ea7f0c8` 等の H3 対応コミット
- [ ] leejet 版 DiT + TE GGUF、両 VAE を配置
- [ ] `--llm_vision` 不要（leejet TE は vision 同梱）
- [ ] FL2VA 生成が成功（56f / 24fps / 864×480 / 音声あり）
- [ ] `git status --short` にモデル・出力が含まれない

---

## 11. 更新履歴

- `2026-08-05` 初版作成（pixi CUDA13 ビルド・leejet GGUF・FL2VA 実機検証 101.02s を記録）

---

## 12. このドキュメントについて

新規参加者が stable-diffusion.cpp による MiniMax-H3 GGUF 生成（特に FL2VA）を短時間で再現・継続できるよう記述しています。上流の仕様詳細は `docs/minimax_h3.md` を、ビルドは `docs/build.md` を参照のこと。
