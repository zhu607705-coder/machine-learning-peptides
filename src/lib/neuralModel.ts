export interface NormalizationStats {
  mean: number[];
  std: number[];
}

export interface RegressionMetrics {
  mae: number;
  rmse: number;
  r2: number;
}

export interface ModelEvaluation {
  purity: RegressionMetrics;
  yield: RegressionMetrics;
  combinedMae: number;
  combinedRmse: number;
}

export interface TrainedPeptideNetwork {
  version: string;
  activation: 'tanh';
  featureNames: string[];
  hiddenSize: number;
  inputNormalization: NormalizationStats;
  targetNormalization: NormalizationStats;
  weights: {
    W1: number[][];
    b1: number[];
    W2: number[][];
    b2: number[];
  };
  training: {
    datasetSeed: number;
    splitSeed: number;
    scenarioCounts: Record<string, number>;
    splitSizes: {
      train: number;
      validation: number;
      test: number;
    };
    bestConfig: {
      hiddenSize: number;
      learningRate: number;
      l2: number;
      maxEpochs: number;
      patience: number;
      bestEpoch: number;
    };
    validation: ModelEvaluation;
    test: ModelEvaluation;
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function normalizeVector(values: number[], stats: NormalizationStats): number[] {
  return values.map((value, index) => (value - stats.mean[index]) / stats.std[index]);
}

export function denormalizeVector(values: number[], stats: NormalizationStats): number[] {
  return values.map((value, index) => (value * stats.std[index]) + stats.mean[index]);
}

export function forwardNormalized(model: TrainedPeptideNetwork, normalizedFeatures: number[]): number[] {
  const hidden = model.weights.b1.map((bias, rowIndex) => {
    let sum = bias;

    for (let columnIndex = 0; columnIndex < normalizedFeatures.length; columnIndex += 1) {
      sum += model.weights.W1[rowIndex][columnIndex] * normalizedFeatures[columnIndex];
    }

    return Math.tanh(sum);
  });

  return model.weights.b2.map((bias, outputIndex) => {
    let sum = bias;

    for (let hiddenIndex = 0; hiddenIndex < hidden.length; hiddenIndex += 1) {
      sum += model.weights.W2[outputIndex][hiddenIndex] * hidden[hiddenIndex];
    }

    return sum;
  });
}

export function predictTargets(model: TrainedPeptideNetwork, rawFeatures: number[]): { purity: number; yield: number } {
  const normalizedInput = normalizeVector(rawFeatures, model.inputNormalization);
  const normalizedOutput = forwardNormalized(model, normalizedInput);
  const [purity, yieldValue] = denormalizeVector(normalizedOutput, model.targetNormalization);

  return {
    purity: clamp(purity, 10, 99.5),
    yield: clamp(yieldValue, 5, 95)
  };
}
