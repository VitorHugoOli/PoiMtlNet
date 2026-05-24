# Questions

- How behaves the B9 model with HGI instead of check2hgi ?

# Considerations

- The previus head experimetns for the check2hig for the next-reg and next-cat was very suspecial, and the resuls
  werent concreted since the fusin and hgi had a better fit with a differnet stl head, I belive THAT WITH THE NEW
  CHECK2HGI we
  should be more riguris an re-try the stl heads for this both tasks at same time we should evalute on how this stls
  head
  are implemented maybe for each task we should have differente variations to best fit they needs , luncha a advisor to
  reoginaze this ideisas and imrpove this test and make somthing more consistente.

- Are the windows sequence of the next-reg been created correctly and the mask in the code been applied correctly ?
- The C3 is a very strong find of our paper but I am a little concern, can you you aduti it and if necessary lunch some
  new experiments over it
- The dataset during the batch has different distribution of the tasks, maybe we should try to balance it more and see
  if it has an effect on the performance of the model, maybe we can try to have a more balanced batch in terms of the
  number of samples for each task, or maybe we can try to have a more balanced batch in terms of the number of samples
  for each class within each task, this can be done by oversampling or undersampling the data for each task or class.
  Think about this problem and evalute.
- About the arch of the MTL, in a previus session we evalute that the original abalation study on the arch of the MTL
  was with a very simple archs variantes of the proposes apporachs like the cross-stitch and other apporachs that you
  can find in the codebase; I belive worht to revised this study in a more regirus way, where we evalute the orginal
  paper and code for the approach and do a real and strong implementation of the apporachs for our usecase, and maybe
  even propose some change to beetter fit to our usecases and iunputs, as we had to do with the crossattn; also in the
  previus ablation we look just for the joint result, maybe worth to look for the two tasks resuslt to have more
  insights
    - Maybe worth try more soft sahred archs
    - Maybe worth try more hard shared archs
    - Maybe worth try more dynamic shared archs
    - Or even archs that uses many apporaches at one like a cross-stitch with a cross-attention or a mmoe with a
      cross-attention and other variations;
    - We need to favor archs that is natural of handles better different inputs for different tasks;
- The mtl_loss soffres with a similar problem as the one we had with the crossattn, also static loss weight are not the
  most simple and has no sofistic logic on it to handle the different loss magnitudes and the different learning speed
  of the different tasks. Maybe we should retry the most advanced approaches for the mtl loss we already did it but we
  coud have a more critical eye over why it hasnt work maybe we haven implemented correct, maybe we dindt get the
  hyperpermstes right we ne d to be more rigures over it
- After the ablation study of the MTL arch and the MTL loss, we can try to combine the best approaches for both and see
  if we can get a better result and re-do the LR schedule / optimezers recipes been more critical and apply the
  knlloadge of the previus studies so we can imporve them for our scenario
- We should pay attention if we are usgin the best formular for The α scalar


- I am not sure in what step we should do it but the mtl heads can be different from the best stl heads, so we need to
  conducting a similiar abalation of heads for the mtl, also not just swaping them but also evalute if we need to do
  some change on them to do a better fit if the mtl arch; the problem is when to fit this after consolided the mtl ? but
  to consolided the mtl we need to have heads? Maybe we should select the best head, improve the mtl and then we do this
  final abalation ?

- We could also work on the `13.2,13.3` that make sens for this study
- Fix the the 13.4 first before any start working
- The 14.4 study is very interesting to do and don't take long
