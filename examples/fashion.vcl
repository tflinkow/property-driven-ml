-- original FashionMNIST image (1 channel) with pixel values in [0, 1]
type UnnormalisedImage = Tensor Real [28, 28]

-- image normalised with mean / std normalisation (common when training computer vision networks)
type NormalisedImage = Tensor Real [28, 28]

type Label = Index 10

-- normalisation:
mean : Real
mean = 0.2860

std : Real
std = 0.3530

-- normalisation: x_norm = (x - mean) / std
normalise : UnnormalisedImage -> NormalisedImage
normalise x =
  foreach h .
    foreach w .
      ( (x ! h ! w) - mean ) / std

-- denormalisation: x = x_norm * std + mean
denormalise : NormalisedImage -> UnnormalisedImage
denormalise x =
  foreach h .
    foreach w .
      ( (x ! h ! w) * std ) + mean

-- pixel values between 0 and 1
validImage : UnnormalisedImage -> Bool
validImage x = forall h w . 0 <= x ! h ! w <= 1

@network
classifier : NormalisedImage -> Tensor Real [10]

@parameter
epsilon : Real

@parameter
delta : Real

boundedByEpsilon : UnnormalisedImage -> Bool
boundedByEpsilon x = forall i j . -epsilon <= x ! i ! j <= epsilon

-- true label has maximal logit
predictsCR : UnnormalisedImage -> Label -> Bool
predictsCR x c =
  forall i .
    i != c =>
      (classifier (normalise x)) ! c >= (classifier (normalise x)) ! i

-- true label greater than threshold
predictsSCR : UnnormalisedImage -> Label -> Bool
predictsSCR x c = (classifier (normalise x)) ! c >= delta

@parameter(infer=True)
n : Nat

@dataset
images : Vector UnnormalisedImage n

@dataset
labels : Vector Label n

cr : UnnormalisedImage -> Label -> Bool
cr image label = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      predictsCR perturbedImage label

scr : UnnormalisedImage -> Label -> Bool
scr image label = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      predictsSCR perturbedImage label

isClothing : Label -> Bool
isClothing c =
  c == 0 or c == 2 or c == 3 or c == 4 or c == 6

isFootwear : Label -> Bool
isFootwear c =
  c == 5 or c == 7 or c == 9

-- true label beats all logits in the opposite semantic group
predictsClothingFootwear : UnnormalisedImage -> Label -> Bool
predictsClothingFootwear x c =
  let y = classifier (normalise x) in
    (isClothing c => (forall i . isFootwear i => y ! c >= y ! i))
    and
    (isFootwear c => (forall i . isClothing i => y ! c >= y ! i))

cf : UnnormalisedImage -> Label -> Bool
cf image label = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      predictsClothingFootwear perturbedImage label

@property
robustCR : Vector Bool n
robustCR = foreach i . cr (images ! i) (labels ! i)

@property
robustSCR : Vector Bool n
robustSCR = foreach i . scr (images ! i) (labels ! i)

@property
clothingFootwear : Vector Bool n
clothingFootwear = foreach i . cf (images ! i) (labels ! i)