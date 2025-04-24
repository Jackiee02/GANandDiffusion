# GANandDiffusion
## Scripts

- **gan_mnist.py**  
  Implements a DCGAN-style training pipeline on MNIST:  
  - Defines `Generator` and `Discriminator` classes with Conv(Transpose)+BatchNorm architectures  
  - Initializes weights, loads and normalizes MNIST, and trains for 50 epochs  
  - Records batch-level and epoch-level losses (with moving average smoothing)  
  - Saves sample grids under `samples/`, model checkpoints (`generator.pth`, `discriminator.pth`), and loss curves (`loss_curves_smoothed_and_epoch.png`)  

- **DiffusionModel.py**  
  Runs high-resolution image synthesis using Stable Diffusion:  
  - Configures the `stabilityai/stable-diffusion-2-1-base` pipeline (FP16, attention slicing, DPMSolverMultistepScheduler)  
  - Specifies positive/negative prompts for a mountaineer-at-sunrise scene  
  - Performs 150 denoising steps and measures inference time  
  - Outputs the final image as `DiffusionPicture.png`  
