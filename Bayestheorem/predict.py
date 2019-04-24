def predict(summaries, inputVector):
  probabilities = calculateClassProbabilities(summaries, inputVector)
  bestLabel, bestProb = None, -1
  for classValue, probability in probabilities.items():
    if bestLabel is None or probability &amp;gt; bestProb:
      bestProb = probability
      bestLabel = classValue
  return bestLabel