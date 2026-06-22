# Diffusion Inpainting Benchmark on COCO val2017

Research-oriented Colab project for object removal and background reconstruction with diffusion inpainting models.

## Research question

How much do model choice, mask dilation, classifier-free guidance and inference steps affect reconstruction quality after removing an object from an image?

## Colab launch order

1. Select a GPU runtime in Google Colab.
2. Run the `Установка` cell once.
3. Use `Runtime -> Restart session`.
4. Continue with `Импорты` and run the notebook from top to bottom.

The dependency versions are pinned deliberately. Do not replace them with unbounded `-U` installs in the same runtime: current Colab images contain tightly coupled CUDA, pandas and requests packages.

The project evaluates three diffusion pipelines:

- `runwayml/stable-diffusion-inpainting`
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- `kandinsky-community/kandinsky-2-2-prior` + `kandinsky-community/kandinsky-2-2-decoder-inpaint`

COCO instance masks define the removed region. The original image remains available as ground truth, so the reconstruction can be evaluated quantitatively.

## Dataset

The benchmark uses COCO 2017 validation data:

- 5,000 original validation images are available in the full mode;
- tasks are generated from official instance annotations, not chosen manually;
- each task contains one image, one object category and one object mask;
- the dataset is split into `tune` and `test` partitions using a fixed seed.

`BENCH_MODE` controls the scale:

- `debug`: 40 tasks;
- `research`: 500 tasks, with 100 for tuning and 400 for test;
- `full`: 5,000 tasks, with 1,000 for tuning and 4,000 for test.

## Experiments

1. Environment preflight and dependency verification.
2. COCO task generation and train/test split.
3. Stable Diffusion 1.5 inpainting parameter tuning on `tune`.
4. Final comparison of SD 1.5 Inpainting, SDXL Inpainting and Kandinsky 2.2 Inpainting on `test`.
5. Category-level error analysis and visual inspection of saved benchmark examples.

## Metrics

- `masked_psnr`: reconstruction error inside the removed region;
- `roi_ssim`: structural similarity on the region around the mask;
- `roi_lpips`: perceptual distance on the region around the mask;
- `outside_mae`: change outside the mask, used as a preservation check;
- runtime per task.

Lower `roi_lpips` and `outside_mae` are better. Higher `masked_psnr` and `roi_ssim` are better.

## Papers

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
- Lugmayr et al., *RePaint: Inpainting Using Denoising Diffusion Probabilistic Models*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html
- Kirillov et al., *Segment Anything*, ICCV 2023: https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html

The notebook uses COCO masks as the inpainting region. SAM is listed as methodological context for promptable mask construction, but it is not required by the benchmark pipeline.

Kandinsky 2.2 is included as an open model from Sber AI. It runs locally in Colab through `diffusers`, so it can be evaluated on the same masks, prompts and random seeds as the Stable Diffusion baselines.
