## Considerations

Check2hig - Inputs
- Can we add more features for the `per-node features` features that will help in the next-reg, next-cat and next-poi
- On the edge weight is worth to do some more advanced so we can capture more nunces of the time like weekend and week, morning, afternoon, night and noon ?
- also on the edge weight, worth try the: Delaunay triangulation over coords, weighted by distance and a cross‑region penalty w_r with the temporal decay ?
- 

Check2hgi - Model
- Isn't the CheckinEncoder very simple ? Can we improve it with a more complex techinic form the literature or even from the poi2vec ?
- The `Check2HGI dropped HGI's 25% hard‑negative mining for an O(1) vs O(R) speedup — see Check2HGIModule.py:189-202` can't this be hurting the peformace? I think worth try back wiht the original apporach and run some experiments
- Idem, for `two‑pass feature‑level corruption` maybe worths to back wiht this feature, no ?

Misselenius
- Maybe on the Checkin2POI we could chagne it to have the fclass as input as well so on this layer we could have a style more like the po2vec and cpture better the region infromations
- On the check2hgi can we add somthing similar to the Delaunay spatial POI‑POI graph ?
- Can we also add a `4th contrastive boundary (implicit)` on the check2 hgi or somthing similar to it that may improve the emebeddings 


Extra
- If we have the poi2vec embedding in the input of the node for the cehck2hgi ?
  - A v2 of this ideia could be creating a strucuture the intead of traing hte pi2vec and add it to the chec2hgi we encapsulate the logic and apporach od the poi2vec in the check2hgi so be more refficient