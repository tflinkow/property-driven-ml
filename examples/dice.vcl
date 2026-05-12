-- original RGB image (3 channels) with pixel values in [0, 1]
type UnnormalisedImage = Tensor Real [3, 28, 28]

-- image normalised with mean / std normalisation (common when training computer vision networks)
type NormalisedImage = Tensor Real [3, 28, 28]

-- normalisation:
mean : Vector Real 3
mean = [0.7232479453086853, 0.7257601618766785, 0.6415771842002869]

std : Vector Real 3
std = [0.32942402362823486, 0.24738003313541412, 0.2831753194332123]

-- normalisation: x_norm = (x - mean) / std
normalise : UnnormalisedImage -> NormalisedImage
normalise x =
  foreach c .
    foreach h .
      foreach w .
        let m = mean ! c in
        let s = std ! c in
          ( (x ! c ! h ! w) - m ) / s

-- denormalisation: x = x_norm * std + mean
denormalise : NormalisedImage -> UnnormalisedImage
denormalise x =
  foreach c .
    foreach h .
      foreach w .
        let m = mean ! c in
        let s = std ! c in
          ( (x ! c ! h ! w) * s ) + m

-- pixel values between 0 and 1
validImage : UnnormalisedImage -> Bool
validImage x = 
  forall c h w . 0 <= x ! c ! h ! w <= 1

@network
classifier : NormalisedImage -> Tensor Real [6]

logit : UnnormalisedImage -> Index 6 -> Real
logit x i = 
  ( classifier (normalise x) ) ! i

-- a label i is predicted if its logit y_i > 0
predicts : UnnormalisedImage -> Index 6 -> Bool
predicts x i = 
  logit x i > 0

@parameter
epsilon : Real

boundedByEpsilon : UnnormalisedImage -> Bool
boundedByEpsilon x = 
  forall i j k . -epsilon <= x ! i ! j ! k <= epsilon

@parameter(infer=True)
n : Nat

@dataset
images : Vector UnnormalisedImage n

@dataset
labels : Vector (Vector (Index 6) 3) n

oppositePairs : Vector (Vector (Index 6) 2) 3
oppositePairs = [ [0, 5], [1, 4], [2, 3] ]

notBothPredicted : UnnormalisedImage -> Index 6 -> Index 6 -> Bool
notBothPredicted image face1 face2 =
  (not (predicts image face1) or not (predicts image face2))

exactlyOnePredicted : UnnormalisedImage -> Index 6 -> Index 6 -> Bool
exactlyOnePredicted image face1 face2 =
  (predicts image face1 or predicts image face2)
  and
  (not (predicts image face1) or not (predicts image face2))

notBoth : UnnormalisedImage -> Bool
notBoth image = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      (forall p.
        let pair = oppositePairs ! p in
          notBothPredicted perturbedImage (pair ! 0) (pair ! 1)
      )

exactlyOnePerPair : UnnormalisedImage -> Bool
exactlyOnePerPair image = forall perturbation .
  let perturbedImage = image + perturbation in
    boundedByEpsilon perturbation and validImage perturbedImage =>
      (forall p .
        let pair = oppositePairs ! p in
          exactlyOnePredicted perturbedImage (pair ! 0) (pair ! 1)
      )

@property
noOppositePair : Vector Bool n
noOppositePair = foreach i . notBoth (images ! i)

@property
exactlyOne : Vector Bool n
exactlyOne = foreach i . exactlyOnePerPair (images ! i)
