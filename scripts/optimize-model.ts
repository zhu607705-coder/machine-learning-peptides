import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AMINO_ACIDS,
  FEATURE_NAMES,
  extractFeatureBundle,
  type PeptideSynthesisParams
} from '../src/lib/peptideCore';
import type {
  ModelEvaluation,
  NormalizationStats,
  RegressionMetrics,
  TrainedPeptideNetwork
} from '../src/lib/neuralModel';

type ScenarioName =
  | 'easy_linear'
  | 'hydrophobic_long'
  | 'steric_rich'
  | 'charged_difficult'
  | 'cysteine_rich'
  | 'cyclized';

interface WeightedOption<T> {
  value: T;
  weight: number;
}

interface ScenarioConfig {
  name: ScenarioName;
  sampleCount: number;
  lengthRange: [number, number];
  aminoWeights: Array<WeightedOption<string>>;
  topologyWeights: Array<WeightedOption<string>>;
  reagentWeights: Array<WeightedOption<string>>;
  solventWeights: Array<WeightedOption<string>>;
  temperatureWeights: Array<WeightedOption<string>>;
  cleavageWeights: Array<WeightedOption<string>>;
  purityBias: number;
  yieldBias: number;
  noiseScale: number;
}

interface ExampleRecord {
  scenario: ScenarioName;
  params: PeptideSynthesisParams;
  features: number[];
  targets: [number, number];
}

interface NormalizedRecord extends ExampleRecord {
  normalizedFeatures: number[];
  normalizedTargets: [number, number];
}

interface NetworkWeights {
  W1: number[][];
  b1: number[];
  W2: number[][];
  b2: number[];
}

interface CandidateConfig {
  hiddenSize: number;
  learningRate: number;
  l2: number;
  maxEpochs: number;
  patience: number;
}

interface CandidateResult {
  config: CandidateConfig;
  bestEpoch: number;
  weights: NetworkWeights;
  validation: ModelEvaluation;
  test: ModelEvaluation;
}

const DATASET_SEED = 20260301;
const SPLIT_SEED = 20260302;
const TRAINING_SEED = 20260303;

const BULKY_POOL = ['Val', 'Ile', 'Thr', 'Pro', 'Phe', 'Trp', 'Tyr', 'Leu'];
const HYDROPHOBIC_POOL = ['Ala', 'Val', 'Ile', 'Leu', 'Phe', 'Trp', 'Tyr', 'Met'];
const CHARGED_POOL = ['Arg', 'His', 'Lys', 'Asp', 'Glu'];
const DIFFICULT_POOL = ['Arg', 'Cys', 'His', 'Asn', 'Gln'];

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomInt(rng: () => number, min: number, max: number): number {
  return Math.floor((rng() * ((max - min) + 1))) + min;
}

function sampleNormal(rng: () => number, mean = 0, std = 1): number {
  const u1 = Math.max(rng(), 1e-7);
  const u2 = Math.max(rng(), 1e-7);
  const mag = Math.sqrt(-2 * Math.log(u1));
  return mean + (std * mag * Math.cos(2 * Math.PI * u2));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function weightMap(defaultWeight: number, overrides: Record<string, number>): Array<WeightedOption<string>> {
  return AMINO_ACIDS.map((aminoAcid) => ({
    value: aminoAcid,
    weight: overrides[aminoAcid] ?? defaultWeight
  }));
}

function sampleWeighted<T>(rng: () => number, options: Array<WeightedOption<T>>): T {
  const total = options.reduce((sum, option) => sum + option.weight, 0);
  let cursor = rng() * total;

  for (const option of options) {
    cursor -= option.weight;

    if (cursor <= 0) {
      return option.value;
    }
  }

  return options[options.length - 1].value;
}

function shuffleInPlace<T>(rng: () => number, values: T[]): void {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
}

function randomResidueFromPool(rng: () => number, pool: string[]): string {
  return pool[randomInt(rng, 0, pool.length - 1)];
}

function ensureResidueCount(rng: () => number, residues: string[], pool: string[], minimumCount: number): void {
  let count = residues.filter((residue) => pool.includes(residue)).length;

  while (count < minimumCount) {
    const replaceIndex = randomInt(rng, 0, residues.length - 1);
    residues[replaceIndex] = randomResidueFromPool(rng, pool);
    count = residues.filter((residue) => pool.includes(residue)).length;
  }
}

function ensureHydrophobicRun(rng: () => number, residues: string[], minimumRun: number): void {
  if (minimumRun <= 1 || residues.length < minimumRun) {
    return;
  }

  const start = randomInt(rng, 0, residues.length - minimumRun);

  for (let offset = 0; offset < minimumRun; offset += 1) {
    residues[start + offset] = randomResidueFromPool(rng, HYDROPHOBIC_POOL);
  }
}

const SCENARIOS: ScenarioConfig[] = [
  {
    name: 'easy_linear',
    sampleCount: 180,
    lengthRange: [4, 8],
    aminoWeights: weightMap(1, {
      Gly: 3.2,
      Ala: 3.0,
      Ser: 2.2,
      Leu: 1.8,
      Gln: 1.6,
      Cys: 0.4,
      Arg: 0.6,
      Trp: 0.6
    }),
    topologyWeights: [{ value: 'Linear', weight: 1 }],
    reagentWeights: [
      { value: 'HATU', weight: 5 },
      { value: 'PyBOP', weight: 3 },
      { value: 'HBTU', weight: 2 },
      { value: 'DIC/Oxyma', weight: 2 }
    ],
    solventWeights: [
      { value: 'DMF', weight: 5 },
      { value: 'NMP', weight: 3 },
      { value: 'DMF/DCM', weight: 1 }
    ],
    temperatureWeights: [
      { value: 'Room Temperature', weight: 6 },
      { value: 'Microwave 75°C', weight: 2 },
      { value: 'Microwave 90°C', weight: 1 }
    ],
    cleavageWeights: [
      { value: '2 hours', weight: 4 },
      { value: '3 hours', weight: 4 },
      { value: '4 hours', weight: 1 }
    ],
    purityBias: 2.2,
    yieldBias: 1.8,
    noiseScale: 1.2
  },
  {
    name: 'hydrophobic_long',
    sampleCount: 180,
    lengthRange: [8, 16],
    aminoWeights: weightMap(0.9, {
      Ala: 2.1,
      Val: 3.0,
      Ile: 2.8,
      Leu: 2.8,
      Phe: 2.4,
      Trp: 1.8,
      Tyr: 1.6,
      Met: 1.8,
      Gly: 0.6,
      Asp: 0.4
    }),
    topologyWeights: [
      { value: 'Linear', weight: 8 },
      { value: 'Head-to-tail cyclization', weight: 2 }
    ],
    reagentWeights: [
      { value: 'HATU', weight: 4 },
      { value: 'PyBOP', weight: 3 },
      { value: 'HBTU', weight: 2 },
      { value: 'DIC/Oxyma', weight: 1 }
    ],
    solventWeights: [
      { value: 'DMF/DCM', weight: 5 },
      { value: 'NMP', weight: 3 },
      { value: 'DMF', weight: 2 }
    ],
    temperatureWeights: [
      { value: 'Microwave 75°C', weight: 5 },
      { value: 'Room Temperature', weight: 3 },
      { value: 'Microwave 90°C', weight: 2 }
    ],
    cleavageWeights: [
      { value: '3 hours', weight: 4 },
      { value: '4 hours', weight: 4 },
      { value: '2 hours', weight: 1 }
    ],
    purityBias: -2.4,
    yieldBias: -2.8,
    noiseScale: 1.8
  },
  {
    name: 'steric_rich',
    sampleCount: 180,
    lengthRange: [7, 12],
    aminoWeights: weightMap(0.8, {
      Val: 2.8,
      Ile: 2.8,
      Thr: 2.6,
      Pro: 2.4,
      Phe: 2.0,
      Tyr: 1.8,
      Gly: 0.6,
      Asp: 0.5,
      Arg: 0.6
    }),
    topologyWeights: [
      { value: 'Linear', weight: 8 },
      { value: 'Head-to-tail cyclization', weight: 1 }
    ],
    reagentWeights: [
      { value: 'HATU', weight: 4 },
      { value: 'PyBOP', weight: 3 },
      { value: 'HBTU', weight: 2 },
      { value: 'DIC/Oxyma', weight: 1 }
    ],
    solventWeights: [
      { value: 'NMP', weight: 4 },
      { value: 'DMF/DCM', weight: 3 },
      { value: 'DMF', weight: 3 }
    ],
    temperatureWeights: [
      { value: 'Microwave 75°C', weight: 5 },
      { value: 'Room Temperature', weight: 3 },
      { value: 'Microwave 90°C', weight: 2 }
    ],
    cleavageWeights: [
      { value: '3 hours', weight: 5 },
      { value: '4 hours', weight: 2 },
      { value: '2 hours', weight: 2 }
    ],
    purityBias: -1.6,
    yieldBias: -1.2,
    noiseScale: 1.6
  },
  {
    name: 'charged_difficult',
    sampleCount: 180,
    lengthRange: [7, 14],
    aminoWeights: weightMap(0.9, {
      Arg: 2.8,
      His: 2.0,
      Lys: 1.8,
      Asp: 2.4,
      Glu: 2.2,
      Asn: 2.0,
      Cys: 1.4,
      Gln: 1.8,
      Val: 0.6
    }),
    topologyWeights: [
      { value: 'Linear', weight: 7 },
      { value: 'Disulfide cyclization', weight: 1 },
      { value: 'Head-to-tail cyclization', weight: 1 }
    ],
    reagentWeights: [
      { value: 'HATU', weight: 4 },
      { value: 'PyBOP', weight: 2 },
      { value: 'HBTU', weight: 3 },
      { value: 'DIC/Oxyma', weight: 1 }
    ],
    solventWeights: [
      { value: 'NMP', weight: 4 },
      { value: 'DMF', weight: 4 },
      { value: 'DMF/DCM', weight: 2 }
    ],
    temperatureWeights: [
      { value: 'Room Temperature', weight: 5 },
      { value: 'Microwave 75°C', weight: 3 },
      { value: 'Microwave 90°C', weight: 1 }
    ],
    cleavageWeights: [
      { value: '3 hours', weight: 4 },
      { value: '4 hours', weight: 4 },
      { value: '2 hours', weight: 1 }
    ],
    purityBias: -2.0,
    yieldBias: -1.8,
    noiseScale: 2.1
  },
  {
    name: 'cysteine_rich',
    sampleCount: 180,
    lengthRange: [6, 12],
    aminoWeights: weightMap(0.8, {
      Cys: 3.2,
      Gly: 1.8,
      Ser: 1.8,
      Ala: 1.4,
      Arg: 1.2,
      His: 1.2,
      Asp: 0.8
    }),
    topologyWeights: [
      { value: 'Disulfide cyclization', weight: 6 },
      { value: 'Linear', weight: 4 }
    ],
    reagentWeights: [
      { value: 'HATU', weight: 5 },
      { value: 'PyBOP', weight: 2 },
      { value: 'HBTU', weight: 2 },
      { value: 'DIC/Oxyma', weight: 1 }
    ],
    solventWeights: [
      { value: 'DMF', weight: 4 },
      { value: 'NMP', weight: 4 },
      { value: 'DMF/DCM', weight: 2 }
    ],
    temperatureWeights: [
      { value: 'Room Temperature', weight: 6 },
      { value: 'Microwave 75°C', weight: 3 },
      { value: 'Microwave 90°C', weight: 1 }
    ],
    cleavageWeights: [
      { value: '3 hours', weight: 4 },
      { value: '4 hours', weight: 4 },
      { value: '2 hours', weight: 2 }
    ],
    purityBias: -2.4,
    yieldBias: -2.0,
    noiseScale: 2.4
  },
  {
    name: 'cyclized',
    sampleCount: 180,
    lengthRange: [5, 10],
    aminoWeights: weightMap(0.9, {
      Gly: 1.8,
      Ala: 1.4,
      Val: 1.8,
      Pro: 1.8,
      Cys: 1.6,
      Asp: 1.4,
      Lys: 1.2,
      Phe: 1.4
    }),
    topologyWeights: [
      { value: 'Head-to-tail cyclization', weight: 6 },
      { value: 'Disulfide cyclization', weight: 3 },
      { value: 'Linear', weight: 1 }
    ],
    reagentWeights: [
      { value: 'HATU', weight: 5 },
      { value: 'PyBOP', weight: 3 },
      { value: 'HBTU', weight: 1 },
      { value: 'DIC/Oxyma', weight: 1 }
    ],
    solventWeights: [
      { value: 'NMP', weight: 5 },
      { value: 'DMF/DCM', weight: 3 },
      { value: 'DMF', weight: 2 }
    ],
    temperatureWeights: [
      { value: 'Room Temperature', weight: 4 },
      { value: 'Microwave 75°C', weight: 4 },
      { value: 'Microwave 90°C', weight: 2 }
    ],
    cleavageWeights: [
      { value: '4 hours', weight: 5 },
      { value: '3 hours', weight: 3 },
      { value: '2 hours', weight: 1 }
    ],
    purityBias: -2.8,
    yieldBias: -3.2,
    noiseScale: 2.0
  }
];

function buildSequence(rng: () => number, scenario: ScenarioConfig, topology: string): string {
  const length = randomInt(rng, scenario.lengthRange[0], scenario.lengthRange[1]);
  const residues = Array.from({ length }, () => sampleWeighted(rng, scenario.aminoWeights));

  if (scenario.name === 'hydrophobic_long') {
    ensureHydrophobicRun(rng, residues, Math.min(4, residues.length));
  }

  if (scenario.name === 'steric_rich') {
    ensureResidueCount(rng, residues, BULKY_POOL, Math.max(2, Math.ceil(length * 0.35)));
  }

  if (scenario.name === 'charged_difficult') {
    ensureResidueCount(rng, residues, CHARGED_POOL, Math.max(2, Math.ceil(length * 0.3)));
    ensureResidueCount(rng, residues, DIFFICULT_POOL, Math.max(2, Math.ceil(length * 0.25)));
  }

  if (scenario.name === 'cysteine_rich' || topology.includes('Disulfide')) {
    ensureResidueCount(rng, residues, ['Cys'], 2);
  }

  if (scenario.name === 'cyclized' && topology.includes('Head-to-tail') && length < 6) {
    residues.push(randomResidueFromPool(rng, ['Gly', 'Ala', 'Val', 'Pro']));
  }

  const cTerminal = rng() > 0.82 ? 'NH2' : 'OH';
  return `H-${residues.join('-')}-${cTerminal}`;
}

function simulateTargets(rng: () => number, scenario: ScenarioConfig, params: PeptideSynthesisParams): [number, number] {
  const { summary } = extractFeatureBundle(params);
  const weakReagent = summary.reagentScore < 0.62;
  const strongReagent = summary.reagentScore > 0.9;
  const strongSolvent = summary.solventScore > 0.8;
  const highTemperature = summary.temperatureScore > 0.95;
  const moderateMicrowave = params.temperature.includes('75');
  const disulfideMismatch = params.topology.includes('Disulfide') && summary.cysCount < 2;
  const shortHeadToTail = params.topology.includes('Head-to-tail') && summary.length < 6;

  let purity =
    91.5
    - (16 * summary.lengthNorm)
    - (11 * summary.bulkyRatio)
    - (13.5 * summary.difficultRatio)
    - (9.5 * summary.hydrophobicRatio)
    - (8 * summary.topologyComplexity)
    - (7 * summary.aspartimideRisk)
    + (12 * summary.reagentScore)
    + (5 * summary.solventScore)
    + (4.5 * summary.temperatureScore)
    + (4.2 * summary.cleavageScore)
    + (5.5 * summary.breakerRatio)
    + (4.5 * summary.sequenceComplexity)
    - (8 * summary.longestHydrophobicRun)
    + (5 * summary.cysCyclizationFit)
    + scenario.purityBias;

  if (summary.bulkyRatio > 0.3 && strongReagent) purity += 4.2;
  if (summary.hydrophobicRatio > 0.45 && strongSolvent) purity += 4.8;
  if (summary.difficultRatio > 0.28 && weakReagent) purity -= 6.5;
  if (moderateMicrowave && summary.bulkyRatio > 0.28) purity += 3.2;
  if (highTemperature && summary.sensitiveRatio > 0.2) purity -= 5.4;
  if (disulfideMismatch) purity -= 12;
  if (shortHeadToTail) purity -= 8.5;
  if (params.topology.includes('Disulfide') && summary.cysCount >= 2 && params.temperature === 'Room Temperature') purity += 2.6;
  if (summary.length > 10 && params.cleavageTime === '4 hours') purity += 1.4;

  purity += sampleNormal(rng, 0, scenario.noiseScale);
  purity = clamp(purity, 28, 99.5);

  let yieldValue =
    71
    + (0.62 * (purity - 70))
    - (13 * summary.lengthNorm)
    - (8.5 * summary.topologyComplexity)
    - (8 * summary.difficultRatio)
    - (6.5 * summary.longestHydrophobicRun)
    + (5.5 * summary.solventScore)
    + (3.5 * summary.cleavageScore)
    + (2.5 * summary.sequenceComplexity)
    + (2 * summary.breakerRatio)
    + scenario.yieldBias;

  if (moderateMicrowave && summary.bulkyRatio > 0.25) yieldValue += 3.3;
  if (highTemperature && (summary.aspartimideRisk + summary.sulfurRatio) > 0.25) yieldValue -= 4.8;
  if (strongReagent && summary.difficultRatio > 0.2) yieldValue += 2.4;
  if (disulfideMismatch) yieldValue -= 10;
  if (shortHeadToTail) yieldValue -= 5.5;

  yieldValue += sampleNormal(rng, 0, scenario.noiseScale * 1.15);
  yieldValue = clamp(yieldValue, 12, 95);
  yieldValue = Math.min(yieldValue, purity + 6);

  return [purity, yieldValue];
}

function createDataset(): ExampleRecord[] {
  const rng = mulberry32(DATASET_SEED);
  const examples: ExampleRecord[] = [];

  for (const scenario of SCENARIOS) {
    for (let sampleIndex = 0; sampleIndex < scenario.sampleCount; sampleIndex += 1) {
      const topology = sampleWeighted(rng, scenario.topologyWeights);
      const params: PeptideSynthesisParams = {
        sequence: buildSequence(rng, scenario, topology),
        topology,
        couplingReagent: sampleWeighted(rng, scenario.reagentWeights),
        solvent: sampleWeighted(rng, scenario.solventWeights),
        temperature: sampleWeighted(rng, scenario.temperatureWeights),
        cleavageTime: sampleWeighted(rng, scenario.cleavageWeights)
      };

      examples.push({
        scenario: scenario.name,
        params,
        features: extractFeatureBundle(params).vector,
        targets: simulateTargets(rng, scenario, params)
      });
    }
  }

  return examples;
}

function stratifiedSplit(examples: ExampleRecord[]): {
  train: ExampleRecord[];
  validation: ExampleRecord[];
  test: ExampleRecord[];
} {
  const rng = mulberry32(SPLIT_SEED);
  const buckets = new Map<ScenarioName, ExampleRecord[]>();

  for (const example of examples) {
    const bucket = buckets.get(example.scenario) ?? [];
    bucket.push(example);
    buckets.set(example.scenario, bucket);
  }

  const train: ExampleRecord[] = [];
  const validation: ExampleRecord[] = [];
  const test: ExampleRecord[] = [];

  for (const scenario of SCENARIOS) {
    const bucket = [...(buckets.get(scenario.name) ?? [])];
    shuffleInPlace(rng, bucket);
    const trainCount = Math.floor(bucket.length * 0.7);
    const validationCount = Math.floor(bucket.length * 0.15);

    train.push(...bucket.slice(0, trainCount));
    validation.push(...bucket.slice(trainCount, trainCount + validationCount));
    test.push(...bucket.slice(trainCount + validationCount));
  }

  shuffleInPlace(rng, train);
  shuffleInPlace(rng, validation);
  shuffleInPlace(rng, test);

  return { train, validation, test };
}

function computeNormalization(vectors: number[][]): NormalizationStats {
  const dimension = vectors[0].length;
  const mean = Array.from({ length: dimension }, () => 0);
  const variance = Array.from({ length: dimension }, () => 0);

  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) {
      mean[index] += vector[index];
    }
  }

  for (let index = 0; index < dimension; index += 1) {
    mean[index] /= vectors.length;
  }

  for (const vector of vectors) {
    for (let index = 0; index < dimension; index += 1) {
      const delta = vector[index] - mean[index];
      variance[index] += delta * delta;
    }
  }

  return {
    mean,
    std: variance.map((value, index) => Math.max(Math.sqrt(value / vectors.length), 1e-6 + (Math.abs(mean[index]) * 1e-6)))
  };
}

function normalizeExamples(
  examples: ExampleRecord[],
  featureStats: NormalizationStats,
  targetStats: NormalizationStats
): NormalizedRecord[] {
  return examples.map((example) => ({
    ...example,
    normalizedFeatures: example.features.map((value, index) => (value - featureStats.mean[index]) / featureStats.std[index]),
    normalizedTargets: example.targets.map((value, index) => (value - targetStats.mean[index]) / targetStats.std[index]) as [number, number]
  }));
}

function createNetwork(inputSize: number, hiddenSize: number, outputSize: number, rng: () => number): NetworkWeights {
  const limit1 = Math.sqrt(6 / (inputSize + hiddenSize));
  const limit2 = Math.sqrt(6 / (hiddenSize + outputSize));

  return {
    W1: Array.from({ length: hiddenSize }, () =>
      Array.from({ length: inputSize }, () => ((rng() * 2) - 1) * limit1)
    ),
    b1: Array.from({ length: hiddenSize }, () => 0),
    W2: Array.from({ length: outputSize }, () =>
      Array.from({ length: hiddenSize }, () => ((rng() * 2) - 1) * limit2)
    ),
    b2: Array.from({ length: outputSize }, () => 0)
  };
}

function cloneNetwork(weights: NetworkWeights): NetworkWeights {
  return {
    W1: weights.W1.map((row) => [...row]),
    b1: [...weights.b1],
    W2: weights.W2.map((row) => [...row]),
    b2: [...weights.b2]
  };
}

function predictNormalized(weights: NetworkWeights, features: number[]): [number, number] {
  const hidden = weights.b1.map((bias, rowIndex) => {
    let sum = bias;

    for (let columnIndex = 0; columnIndex < features.length; columnIndex += 1) {
      sum += weights.W1[rowIndex][columnIndex] * features[columnIndex];
    }

    return Math.tanh(sum);
  });

  const output = weights.b2.map((bias, outputIndex) => {
    let sum = bias;

    for (let hiddenIndex = 0; hiddenIndex < hidden.length; hiddenIndex += 1) {
      sum += weights.W2[outputIndex][hiddenIndex] * hidden[hiddenIndex];
    }

    return sum;
  });

  return [output[0], output[1]];
}

function denormalizeTarget(values: [number, number], targetStats: NormalizationStats): [number, number] {
  return [
    (values[0] * targetStats.std[0]) + targetStats.mean[0],
    (values[1] * targetStats.std[1]) + targetStats.mean[1]
  ];
}

function computeRegressionMetrics(actual: number[], predicted: number[]): RegressionMetrics {
  const count = actual.length;
  const mean = actual.reduce((sum, value) => sum + value, 0) / count;

  let absoluteError = 0;
  let squaredError = 0;
  let totalSquaredError = 0;

  for (let index = 0; index < count; index += 1) {
    const error = predicted[index] - actual[index];
    absoluteError += Math.abs(error);
    squaredError += error * error;

    const centered = actual[index] - mean;
    totalSquaredError += centered * centered;
  }

  return {
    mae: absoluteError / count,
    rmse: Math.sqrt(squaredError / count),
    r2: totalSquaredError === 0 ? 1 : 1 - (squaredError / totalSquaredError)
  };
}

function evaluate(weights: NetworkWeights, dataset: NormalizedRecord[], targetStats: NormalizationStats): ModelEvaluation {
  const purityActual: number[] = [];
  const purityPredicted: number[] = [];
  const yieldActual: number[] = [];
  const yieldPredicted: number[] = [];

  for (const record of dataset) {
    const prediction = denormalizeTarget(predictNormalized(weights, record.normalizedFeatures), targetStats);
    purityActual.push(record.targets[0]);
    purityPredicted.push(prediction[0]);
    yieldActual.push(record.targets[1]);
    yieldPredicted.push(prediction[1]);
  }

  const purity = computeRegressionMetrics(purityActual, purityPredicted);
  const yieldMetric = computeRegressionMetrics(yieldActual, yieldPredicted);

  return {
    purity,
    yield: yieldMetric,
    combinedMae: (purity.mae + yieldMetric.mae) / 2,
    combinedRmse: (purity.rmse + yieldMetric.rmse) / 2
  };
}

function trainCandidate(
  config: CandidateConfig,
  trainSet: NormalizedRecord[],
  validationSet: NormalizedRecord[],
  testSet: NormalizedRecord[],
  targetStats: NormalizationStats,
  seedOffset: number
): CandidateResult {
  const rng = mulberry32(TRAINING_SEED + seedOffset);
  const inputSize = trainSet[0].normalizedFeatures.length;
  const outputSize = trainSet[0].normalizedTargets.length;
  const weights = createNetwork(inputSize, config.hiddenSize, outputSize, rng);
  let bestWeights = cloneNetwork(weights);
  let bestEpoch = 0;
  let bestScore = Number.POSITIVE_INFINITY;
  let staleEpochs = 0;

  for (let epoch = 1; epoch <= config.maxEpochs; epoch += 1) {
    const gradW1 = weights.W1.map((row) => row.map(() => 0));
    const gradb1 = weights.b1.map(() => 0);
    const gradW2 = weights.W2.map((row) => row.map(() => 0));
    const gradb2 = weights.b2.map(() => 0);

    for (const record of trainSet) {
      const hiddenRaw = weights.b1.map((bias, rowIndex) => {
        let sum = bias;

        for (let columnIndex = 0; columnIndex < inputSize; columnIndex += 1) {
          sum += weights.W1[rowIndex][columnIndex] * record.normalizedFeatures[columnIndex];
        }

        return sum;
      });

      const hidden = hiddenRaw.map((value) => Math.tanh(value));
      const output = weights.b2.map((bias, outputIndex) => {
        let sum = bias;

        for (let hiddenIndex = 0; hiddenIndex < config.hiddenSize; hiddenIndex += 1) {
          sum += weights.W2[outputIndex][hiddenIndex] * hidden[hiddenIndex];
        }

        return sum;
      });

      const dOutput = output.map((value, outputIndex) => {
        const error = value - record.normalizedTargets[outputIndex];
        return error / outputSize;
      });

      for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
        gradb2[outputIndex] += dOutput[outputIndex];

        for (let hiddenIndex = 0; hiddenIndex < config.hiddenSize; hiddenIndex += 1) {
          gradW2[outputIndex][hiddenIndex] += dOutput[outputIndex] * hidden[hiddenIndex];
        }
      }

      const dHidden = hidden.map((hiddenValue, hiddenIndex) => {
        let gradient = 0;

        for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
          gradient += dOutput[outputIndex] * weights.W2[outputIndex][hiddenIndex];
        }

        return gradient * (1 - (hiddenValue * hiddenValue));
      });

      for (let hiddenIndex = 0; hiddenIndex < config.hiddenSize; hiddenIndex += 1) {
        gradb1[hiddenIndex] += dHidden[hiddenIndex];

        for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
          gradW1[hiddenIndex][inputIndex] += dHidden[hiddenIndex] * record.normalizedFeatures[inputIndex];
        }
      }
    }

    const learningRate = config.learningRate / trainSet.length;

    for (let hiddenIndex = 0; hiddenIndex < config.hiddenSize; hiddenIndex += 1) {
      weights.b1[hiddenIndex] -= learningRate * gradb1[hiddenIndex];

      for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
        const l2Penalty = config.l2 * weights.W1[hiddenIndex][inputIndex];
        weights.W1[hiddenIndex][inputIndex] -= learningRate * (gradW1[hiddenIndex][inputIndex] + l2Penalty);
      }
    }

    for (let outputIndex = 0; outputIndex < outputSize; outputIndex += 1) {
      weights.b2[outputIndex] -= learningRate * gradb2[outputIndex];

      for (let hiddenIndex = 0; hiddenIndex < config.hiddenSize; hiddenIndex += 1) {
        const l2Penalty = config.l2 * weights.W2[outputIndex][hiddenIndex];
        weights.W2[outputIndex][hiddenIndex] -= learningRate * (gradW2[outputIndex][hiddenIndex] + l2Penalty);
      }
    }

    const validationMetrics = evaluate(weights, validationSet, targetStats);

    if (validationMetrics.combinedRmse + 1e-5 < bestScore) {
      bestScore = validationMetrics.combinedRmse;
      bestWeights = cloneNetwork(weights);
      bestEpoch = epoch;
      staleEpochs = 0;
    } else {
      staleEpochs += 1;
    }

    if (staleEpochs >= config.patience) {
      break;
    }
  }

  return {
    config,
    bestEpoch,
    weights: bestWeights,
    validation: evaluate(bestWeights, validationSet, targetStats),
    test: evaluate(bestWeights, testSet, targetStats)
  };
}

function writeArtifacts(model: TrainedPeptideNetwork): void {
  const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const modelPath = path.join(rootDir, 'src', 'lib', 'modelArtifacts.ts');
  const reportDir = path.join(rootDir, 'artifacts');
  const reportPath = path.join(reportDir, 'peptide-model-report.json');
  const jsonModelPath = path.join(reportDir, 'peptide-model.json');

  mkdirSync(reportDir, { recursive: true });

  const source = `import type { TrainedPeptideNetwork } from './neuralModel';

export const trainedPeptideNetwork: TrainedPeptideNetwork = ${JSON.stringify(model, null, 2)};

export default trainedPeptideNetwork;
`;

  writeFileSync(modelPath, source, 'utf8');
  writeFileSync(jsonModelPath, `${JSON.stringify(model, null, 2)}\n`, 'utf8');
  writeFileSync(reportPath, `${JSON.stringify(model.training, null, 2)}\n`, 'utf8');
}

function main(): void {
  const dataset = createDataset();
  const { train, validation, test } = stratifiedSplit(dataset);
  const featureStats = computeNormalization(train.map((record) => record.features));
  const targetStats = computeNormalization(train.map((record) => [...record.targets]));
  const trainSet = normalizeExamples(train, featureStats, targetStats);
  const validationSet = normalizeExamples(validation, featureStats, targetStats);
  const testSet = normalizeExamples(test, featureStats, targetStats);
  const candidates: CandidateConfig[] = [
    { hiddenSize: 8, learningRate: 0.08, l2: 0.0005, maxEpochs: 420, patience: 55 },
    { hiddenSize: 8, learningRate: 0.12, l2: 0.0015, maxEpochs: 420, patience: 55 },
    { hiddenSize: 12, learningRate: 0.08, l2: 0.0005, maxEpochs: 480, patience: 60 },
    { hiddenSize: 12, learningRate: 0.12, l2: 0.0015, maxEpochs: 480, patience: 60 },
    { hiddenSize: 16, learningRate: 0.06, l2: 0.0005, maxEpochs: 520, patience: 65 },
    { hiddenSize: 16, learningRate: 0.09, l2: 0.0015, maxEpochs: 520, patience: 65 }
  ];

  const results = candidates.map((candidate, index) =>
    trainCandidate(candidate, trainSet, validationSet, testSet, targetStats, index)
  );

  results.sort((left, right) => left.validation.combinedRmse - right.validation.combinedRmse);
  const best = results[0];

  const model: TrainedPeptideNetwork = {
    version: 'synthetic-gridsearch-v2',
    activation: 'tanh',
    featureNames: FEATURE_NAMES,
    hiddenSize: best.config.hiddenSize,
    inputNormalization: featureStats,
    targetNormalization: targetStats,
    weights: best.weights,
    training: {
      datasetSeed: DATASET_SEED,
      splitSeed: SPLIT_SEED,
      scenarioCounts: Object.fromEntries(SCENARIOS.map((scenario) => [scenario.name, scenario.sampleCount])),
      splitSizes: {
        train: train.length,
        validation: validation.length,
        test: test.length
      },
      bestConfig: {
        ...best.config,
        bestEpoch: best.bestEpoch
      },
      validation: best.validation,
      test: best.test
    }
  };

  writeArtifacts(model);

  console.log('Synthetic dataset size:', dataset.length);
  console.log('Train/Validation/Test:', train.length, validation.length, test.length);
  console.log('Best config:', model.training.bestConfig);
  console.log('Validation RMSE:', {
    purity: model.training.validation.purity.rmse.toFixed(3),
    yield: model.training.validation.yield.rmse.toFixed(3),
    combined: model.training.validation.combinedRmse.toFixed(3)
  });
  console.log('Test RMSE:', {
    purity: model.training.test.purity.rmse.toFixed(3),
    yield: model.training.test.yield.rmse.toFixed(3),
    combined: model.training.test.combinedRmse.toFixed(3)
  });
}

main();
