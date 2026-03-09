export interface PeptideSynthesisParams {
  sequence: string;
  topology: string;
  couplingReagent: string;
  solvent: string;
  temperature: string;
  cleavageTime: string;
}

export interface FeatureSummary {
  length: number;
  lengthNorm: number;
  bulkyRatio: number;
  difficultRatio: number;
  hydrophobicRatio: number;
  chargedRatio: number;
  breakerRatio: number;
  sulfurRatio: number;
  aspartimideRisk: number;
  reagentScore: number;
  topologyComplexity: number;
  temperatureScore: number;
  solventScore: number;
  cleavageScore: number;
  sequenceComplexity: number;
  longestHydrophobicRun: number;
  cysCyclizationFit: number;
  cysCount: number;
  sensitiveRatio: number;
}

export interface FeatureBundle {
  residues: string[];
  featureNames: string[];
  vector: number[];
  summary: FeatureSummary;
}

export const AAMass: Record<string, number> = {
  Gly: 57.02146,
  Ala: 71.03711,
  Ser: 87.03203,
  Pro: 97.05276,
  Val: 99.06841,
  Thr: 101.04768,
  Cys: 103.00919,
  Ile: 113.08406,
  Leu: 113.08406,
  Asn: 114.04293,
  Asp: 115.02694,
  Gln: 128.05858,
  Lys: 128.09496,
  Glu: 129.04259,
  Met: 131.04049,
  His: 137.05891,
  Phe: 147.06841,
  Arg: 156.10111,
  Tyr: 163.06333,
  Trp: 186.07931
};

export const AMINO_ACIDS = Object.keys(AAMass);

export const FEATURE_NAMES = [
  'length_norm',
  'bulky_ratio',
  'difficult_ratio',
  'hydrophobic_ratio',
  'charged_ratio',
  'breaker_ratio',
  'sulfur_ratio',
  'aspartimide_risk',
  'reagent_score',
  'topology_complexity',
  'temperature_score',
  'solvent_score',
  'cleavage_score',
  'sequence_complexity',
  'longest_hydrophobic_run',
  'cys_cyclization_fit'
];

const TERMINAL_GROUPS = new Set(['H', 'OH', 'NH2']);
const BULKY_RESIDUES = new Set(['Val', 'Ile', 'Thr', 'Pro', 'Phe', 'Trp', 'Tyr', 'Leu']);
const DIFFICULT_RESIDUES = new Set(['Arg', 'Cys', 'His', 'Asn', 'Gln']);
const HYDROPHOBIC_RESIDUES = new Set(['Ala', 'Val', 'Ile', 'Leu', 'Phe', 'Trp', 'Tyr', 'Met', 'Pro']);
const CHARGED_RESIDUES = new Set(['Asp', 'Glu', 'Lys', 'Arg', 'His']);
const BREAKER_RESIDUES = new Set(['Gly', 'Pro']);
const SULFUR_RESIDUES = new Set(['Cys', 'Met']);
const ASPARTIMIDE_RESIDUES = new Set(['Asp', 'Asn']);
const SENSITIVE_RESIDUES = new Set(['Asp', 'Asn', 'Cys', 'Met']);

function ratio(count: number, total: number): number {
  return total === 0 ? 0 : count / total;
}

function countResidues(residues: string[], target: Set<string>): number {
  return residues.reduce((sum, residue) => sum + (target.has(residue) ? 1 : 0), 0);
}

function longestRun(residues: string[], predicate: (residue: string) => boolean): number {
  let best = 0;
  let current = 0;

  for (const residue of residues) {
    if (predicate(residue)) {
      current += 1;
      best = Math.max(best, current);
    } else {
      current = 0;
    }
  }

  return best;
}

export function parsePeptideSequence(sequence: string): string[] {
  return sequence
    .split('-')
    .map((part) => part.trim())
    .filter((part) => part.length > 0 && !TERMINAL_GROUPS.has(part));
}

export function reagentScore(reagent: string): number {
  if (reagent.includes('HATU')) return 1.0;
  if (reagent.includes('DIC/Oxyma')) return 0.9;
  if (reagent.includes('PyBOP')) return 0.84;
  if (reagent.includes('HBTU/DIEA')) return 0.66;
  if (reagent.includes('HBTU')) return 0.66;
  return 0.5;
}

export function topologyComplexity(topology: string): number {
  if (topology.includes('Head-to-tail') || topology.includes('头尾环化')) return 0.95;
  if (topology.includes('Disulfide') || topology.includes('二硫键环化')) return 0.76;
  return 0.12;
}

export function temperatureScore(temperature: string): number {
  if (temperature.includes('90')) return 1.0;
  if (temperature.includes('75')) return 0.72;
  return 0.25;
}

export function solventScore(solvent: string): number {
  if (solvent.includes('NMP')) return 0.86;
  if (solvent.includes('DMF/DCM')) return 0.74;
  return 0.55;
}

export function cleavageScore(cleavageTime: string): number {
  if (cleavageTime.includes('4')) return 1.0;
  if (cleavageTime.includes('3')) return 0.72;
  return 0.44;
}

export function extractFeatureBundle(params: PeptideSynthesisParams): FeatureBundle {
  const residues = parsePeptideSequence(params.sequence);
  const length = Math.max(1, residues.length);

  const bulkyRatio = ratio(countResidues(residues, BULKY_RESIDUES), length);
  const difficultRatio = ratio(countResidues(residues, DIFFICULT_RESIDUES), length);
  const hydrophobicRatio = ratio(countResidues(residues, HYDROPHOBIC_RESIDUES), length);
  const chargedRatio = ratio(countResidues(residues, CHARGED_RESIDUES), length);
  const breakerRatio = ratio(countResidues(residues, BREAKER_RESIDUES), length);
  const sulfurRatio = ratio(countResidues(residues, SULFUR_RESIDUES), length);
  const aspartimideRisk = ratio(countResidues(residues, ASPARTIMIDE_RESIDUES), length);
  const cysCount = residues.filter((residue) => residue === 'Cys').length;
  const sequenceComplexity = new Set(residues).size / length;
  const longestHydrophobicRun = longestRun(residues, (residue) => HYDROPHOBIC_RESIDUES.has(residue)) / length;
  const reagentStrength = reagentScore(params.couplingReagent);
  const topologyScore = topologyComplexity(params.topology);
  const thermalScore = temperatureScore(params.temperature);
  const solventStrength = solventScore(params.solvent);
  const cleavageStrength = cleavageScore(params.cleavageTime);
  const cysCyclizationFit = params.topology.includes('Disulfide') && cysCount >= 2 ? Math.min(1, cysCount / 4) : 0;
  const sensitiveRatio = ratio(countResidues(residues, SENSITIVE_RESIDUES), length);
  const lengthNorm = Math.min(1, Math.max(0, (length - 4) / 12));

  const summary: FeatureSummary = {
    length,
    lengthNorm,
    bulkyRatio,
    difficultRatio,
    hydrophobicRatio,
    chargedRatio,
    breakerRatio,
    sulfurRatio,
    aspartimideRisk,
    reagentScore: reagentStrength,
    topologyComplexity: topologyScore,
    temperatureScore: thermalScore,
    solventScore: solventStrength,
    cleavageScore: cleavageStrength,
    sequenceComplexity,
    longestHydrophobicRun,
    cysCyclizationFit,
    cysCount,
    sensitiveRatio
  };

  const vector = [
    summary.lengthNorm,
    summary.bulkyRatio,
    summary.difficultRatio,
    summary.hydrophobicRatio,
    summary.chargedRatio,
    summary.breakerRatio,
    summary.sulfurRatio,
    summary.aspartimideRisk,
    summary.reagentScore,
    summary.topologyComplexity,
    summary.temperatureScore,
    summary.solventScore,
    summary.cleavageScore,
    summary.sequenceComplexity,
    summary.longestHydrophobicRun,
    summary.cysCyclizationFit
  ];

  return {
    residues,
    featureNames: FEATURE_NAMES,
    vector,
    summary
  };
}

export function calculateExactMassMH(sequence: string, topology: string): number {
  const residues = parsePeptideSequence(sequence);
  let exactMass = 18.01528;

  for (const residue of residues) {
    exactMass += AAMass[residue] ?? 0;
  }

  if (topology.includes('Head-to-tail') || topology.includes('头尾环化')) {
    exactMass -= 18.01528;
  }

  if ((topology.includes('Disulfide') || topology.includes('二硫键环化')) && residues.filter((residue) => residue === 'Cys').length >= 2) {
    exactMass -= 2.016;
  }

  return exactMass + 1.007276;
}
