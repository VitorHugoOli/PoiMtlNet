# Questions

- How behaves the B9 model with HGI instead of check2hgi

# Considerations

- The previus head experimetns for the check2hig for the next-reg and next-cat was very supericial, and the resuls
  werent concreted since the fusin and hgi wach had a better fit with a stl head i BELIVE THAT WITH THE NEW CHECK2HGI we
  should be more riguris an re-try the stl heads for this both head at same time we should evalute on how this stls head
  are implemented maybe for each task we should have differente variations to best fit they need , luncha a dvisor to
  roginaze this ideisas and imrpove this test and make somthing more consistente.

- Are the windows sequence of the next-reg been created correctly and the mask in the code been applied correctly ?
- The C3 is a very strong find of our paper but I am a little concern, can you you aduti it and if necessary lunch some
  new experiments over it
- The dataset duriong the batch has different distribution of the tasks, maybe we should try to balance it more and see
  if it has an effect on the performance of the model, maybe we can try to have a more balanced batch in terms of the
  number of samples for each task, or maybe we can try to have a more balanced batch in terms of the number of samples
  for each class within each task, this can be done by oversampling or undersampling the data for each task or class.
  Think about this problem and evalute.
- About the arch of the MTL, in a previus session we evalute that the original abalation study on the arch of the MTL
  was with a very simple archs variantes of the proposes apporachs like the cross-stitch and other apporachs that you
  can find in the codebase; I belive worht to revised this study in a more regirus way, where we evalute the orginal
  paper and code for the approach a do a real and strong iomplementation of the apporachs for our usecase, and maybe
  even propose some change to beetter fit to our usecases and iunputs, as we had to do with the crossattn; also in the
  previus ablation we look just for the joint result, maybe worth to look for the two tasks resuslt to have more
  insights
    - Maybe worth try more soft sahred archs
    - Maybe worth try more hard shared archs
    - Maybe worth try more dynamic shared archs
    - Or even archs that uses many apporaches at one like a cross-stitch with a cross-attention or a mmoe with a
      cross-attention and other variations;
- The mtl_loss soffres with a similar problem as the one we had with the crossattn, also static loss weight are not the
  most simple and has no sofistic logic on it to handle the different loss magnitudes and the different learning speed
  of the different tasks. Maybe we should retry the most advanced approaches for the mtl loss we already did it but we
  coud have a more critical eye over why it hasnt work maybe we haven timplemented correct, maybe we dindt get the
  hyperpermstes right we ne d to be more rigures over it
- After the ablation study of the MTL arch and the MTL loss, we can try to combine the best approaches for both and see
  if we can get a better result and re-do the LR schedule / optimezers recipes been more critical and apply the
  knlloadge of the previus studies so we can imporve them for our scenario
- We should pay attention if we are usgin the best formular for The α scalar
- 
- 
- A melhor cabeca muda com o mtl