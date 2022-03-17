# Prior_Net

* To add structuredness in the night segmentation models (DA or any) 
* End to End -- Like a auto encoder but on the label space....this is trying to tackle on the label distrubution...such that we are not focusing on style but on the structure as much as possible ...or Unet..will decide that
- [ ] Reading out autoencoder, finding the best archi for the for this task. ..better then UNet cause here we are not segmentation but we are doing some sort of image denoising but on the label space ...either 3 channel or the main 19 one (containing overall probability distribution)
- [ ] Augmentation -- what to use ( any random black out of the intial RGB images...and then taking predictions from the Same model
- [ ] Can train the different datasets for pretraining the prior net and the fine tune on the ACDC dataset ... or anyother strategies (if suituable)
- [ ] Two exps to do: 
     - [ ] 19 to 19 
     - [ ] 3 to 3
- [ ] Check with other Model as well...like MGCDA model because where they are clear structural errors 
- [ ] Can also train prior-net with  other model output simultaneousely as well...i.e. with mgcda and dannet output label...and see its performance improvement 
- [ ] Check on Saliency map, attention map or entropy any other is required or not. 
- [ ] can see error correction ocr model paper from rohit sir 
- [ ] After training (externally) this prior and then you can train it with the original model from overall end to end stage (i.e. outer loop end to end first and then the overall end to end) and try to get a significant boost. 
