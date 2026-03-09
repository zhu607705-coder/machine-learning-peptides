import { trainedPeptideNetwork } from './modelArtifacts';
import { predictTargets } from './neuralModel';
import {
  AAMass,
  calculateExactMassMH,
  extractFeatureBundle,
  type PeptideSynthesisParams
} from './peptideCore';

interface MassPeak {
  mz: number;
  abundance: number;
  label: string;
  source: 'predicted' | 'literature';
  reference?: string;
}

interface PredictedByproduct {
  name: string;
  mz: number;
  cause: string;
}

interface LiteraturePeak {
  mz: number;
  label: string;
  description: string;
  condition: string;
  reference: string;
}

// Literature-based common peptide mass spectrometry peak positions
// Sources: 
// 1. Steen & Mann (2004) Nature Reviews Molecular Cell Biology 5, 699-711
// 2. Domon & Aebersold (2006) Science 312, 212-217
// 3. Common fragmentation patterns in ESI-MS/MS of peptides
const LITERATURE_PEAKS: Record<string, LiteraturePeak[]> = {
  // Common adducts and modifications
  adducts: [
    { mz: 22.989, label: '[M+Na]+', description: '钠加合峰', condition: '常见背景离子', reference: 'Steen & Mann, 2004' },
    { mz: 38.963, label: '[M+K]+', description: '钾加合峰', condition: '常见背景离子', reference: 'Steen & Mann, 2004' },
    { mz: 42.011, label: '[M+ACN+H]+', description: '乙腈加合峰', condition: '体系存在乙腈时', reference: 'LC-MS 常见现象' }
  ],
  // Common neutral losses
  neutralLosses: [
    { mz: -18.011, label: '[M-H2O+H]+', description: '失水峰', condition: '含 Ser/Thr/Asp/Glu 时更常见', reference: 'Domon & Aebersold, 2006' },
    { mz: -17.027, label: '[M-NH3+H]+', description: '失氨峰', condition: '含 Asn/Gln/Arg 时更常见', reference: 'Domon & Aebersold, 2006' },
    { mz: -48.003, label: '[M-H2O-CO2+H]+', description: '失水并失二氧化碳峰', condition: 'C 端含 Asp/Glu 时更常见', reference: 'Steen & Mann, 2004' }
  ],
  // Common fragment ions (b-ions and y-ions)
  fragmentIons: [
    { mz: 0.0, label: 'b 离子系列', description: 'N 端碎片离子', condition: 'MS/MS 裂解时出现', reference: 'Roepstorff & Fohlman, 1984' },
    { mz: 0.0, label: 'y 离子系列', description: 'C 端碎片离子', condition: 'MS/MS 裂解时出现', reference: 'Roepstorff & Fohlman, 1984' }
  ],
  // Common synthesis-related byproducts
  synthesisByproducts: [
    { mz: -56.026, label: '[M-tBu+H]+', description: 'tBu 保护基脱落相关峰', condition: '含 tBu 保护残基时', reference: 'SPPS 标准经验' },
    { mz: -100.016, label: '[M-Boc+H]+', description: 'Boc 保护基脱落相关峰', condition: '含 Boc 保护残基时', reference: 'SPPS 标准经验' },
    { mz: -120.032, label: '[M-Fmoc+H]+', description: 'Fmoc 残留或脱落相关峰', condition: '脱保护不完全时', reference: 'SPPS 标准经验' }
  ]
};

// Amino acid immonium ions (common in MS/MS)
// Source: Papayannopoulos (1995) Mass Spectrometry Reviews 14, 47-73
const IMMONIUM_IONS: Record<string, { mz: number; residue: string }> = {
  Gly: { mz: 30.034, residue: 'Gly' },
  Ala: { mz: 44.050, residue: 'Ala' },
  Ser: { mz: 60.045, residue: 'Ser' },
  Pro: { mz: 70.065, residue: 'Pro' },
  Val: { mz: 72.081, residue: 'Val' },
  Thr: { mz: 74.060, residue: 'Thr' },
  Cys: { mz: 76.022, residue: 'Cys' },
  Ile: { mz: 86.097, residue: 'Ile' },
  Leu: { mz: 86.097, residue: 'Leu' },
  Asn: { mz: 87.055, residue: 'Asn' },
  Asp: { mz: 88.039, residue: 'Asp' },
  Gln: { mz: 101.071, residue: 'Gln' },
  Lys: { mz: 101.107, residue: 'Lys' },
  Glu: { mz: 102.055, residue: 'Glu' },
  Met: { mz: 104.053, residue: 'Met' },
  His: { mz: 110.071, residue: 'His' },
  Phe: { mz: 120.081, residue: 'Phe' },
  Arg: { mz: 129.113, residue: 'Arg' },
  Tyr: { mz: 136.076, residue: 'Tyr' },
  Trp: { mz: 159.092, residue: 'Trp' }
};

function hashSeed(input: string): number {
  let hash = 2166136261;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;

  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function findFirstMatchingResidue(sequence: string[], target: Set<string>): string | null {
  return sequence.find((residue) => target.has(residue)) ?? null;
}

// Generate b-ion and y-ion series for a peptide
function generateFragmentIons(sequence: string[], exactMassMH: number): MassPeak[] {
  const peaks: MassPeak[] = [];
  let bIonMass = 1.0078; // H+ mass
  let yIonMass = 18.0153 + 1.0078; // H2O + H+ (for full y-ion)
  
  // Calculate total mass
  const totalResidueMass = sequence.reduce((sum, residue) => sum + (AAMass[residue] ?? 0), 0);
  
  // Generate b-ions (N-terminal fragments)
  for (let i = 0; i < sequence.length - 1; i++) {
    bIonMass += AAMass[sequence[i]] ?? 0;
    peaks.push({
      mz: bIonMass,
      abundance: 15 + Math.random() * 25, // Random abundance between 15-40%
      label: `b${i + 1}`,
      source: 'literature',
      reference: 'Roepstorff & Fohlman (1984); Steen & Mann (2004)'
    });
  }
  
  // Generate y-ions (C-terminal fragments)
  for (let i = sequence.length - 1; i > 0; i--) {
    yIonMass += AAMass[sequence[i]] ?? 0;
    peaks.push({
      mz: exactMassMH - bIonMass + AAMass[sequence[i]] + 18.0153 + 1.0078,
      abundance: 20 + Math.random() * 30, // Random abundance between 20-50%
      label: `y${sequence.length - i}`,
      source: 'literature',
      reference: 'Roepstorff & Fohlman (1984); Steen & Mann (2004)'
    });
  }
  
  return peaks;
}

export class LocalPeptideModel {
  public static getModelDiagnostics() {
    return trainedPeptideNetwork.training;
  }

  public static async predict(params: PeptideSynthesisParams): Promise<{
    purity: number;
    yield: number;
    exactMass: number;
    massSpectrum: MassPeak[];
    byproducts: PredictedByproduct[];
    optimizations: string[];
    literaturePeaks: LiteraturePeak[];
  }> {
    await new Promise((resolve) => setTimeout(resolve, 900));

    const featureBundle = extractFeatureBundle(params);
    const rawPrediction = predictTargets(trainedPeptideNetwork, featureBundle.vector);
    const purity = Math.min(99.5, Math.max(10, rawPrediction.purity));
    const predictedYield = Math.min(95, Math.max(5, Math.min(rawPrediction.yield, purity + 6)));
    const exactMassMH = calculateExactMassMH(params.sequence, params.topology);
    const random = createSeededRandom(hashSeed(JSON.stringify(params)));
    const byproducts: PredictedByproduct[] = [];
    const optimizations: string[] = [];
    
    // Main peaks (predicted)
    const massSpectrum: MassPeak[] = [
      {
        mz: exactMassMH, 
        abundance: 100, 
        label: '[M+H]+（目标主峰）',
        source: 'predicted'
      },
      { 
        mz: exactMassMH + 21.9819, 
        abundance: 10 + (random() * 10), 
        label: '[M+Na]+',
        source: 'predicted',
        reference: 'Steen & Mann (2004) Nature Rev Mol Cell Biol 5:699-711'
      },
      { 
        mz: (exactMassMH / 2) + 0.5, 
        abundance: 24 + (random() * 16), 
        label: '[M+2H]2+',
        source: 'predicted',
        reference: 'ESI-MS 中常见的二价带电离子'
      }
    ];

    const residues = featureBundle.residues;
    const summary = featureBundle.summary;
    
    // Add fragment ions (literature-based)
    const fragmentIons = generateFragmentIons(residues, exactMassMH);
    massSpectrum.push(...fragmentIons);
    
    // Add immonium ions for present residues
    const presentImmoniumIons = new Set<string>();
    residues.forEach(residue => {
      if (IMMONIUM_IONS[residue] && !presentImmoniumIons.has(residue)) {
        presentImmoniumIons.add(residue);
        massSpectrum.push({
          mz: IMMONIUM_IONS[residue].mz,
          abundance: 5 + random() * 15,
          label: `${residue} 亚胺离子`,
          source: 'literature',
          reference: 'Papayannopoulos (1995) Mass Spec Rev 14:47-73'
        });
      }
    });

    const bulkyResidue = findFirstMatchingResidue(
      residues,
      new Set(['Val', 'Ile', 'Thr', 'Pro', 'Phe', 'Trp', 'Tyr', 'Leu'])
    );

    if (summary.bulkyRatio > 0.28 && bulkyResidue) {
      const deletionMass = exactMassMH - (AAMass[bulkyResidue] ?? 100);
      byproducts.push({
        name: `${bulkyResidue} 缺失截短肽`,
        mz: deletionMass,
        cause: `${bulkyResidue} 周围的位阻拥挤可能导致偶联不完全，从而留下截短序列。`
      });
      massSpectrum.push({
        mz: deletionMass,
        abundance: Math.max(6, 42 - (purity * 0.28)),
        label: `[M-${bulkyResidue}+H]+`,
        source: 'predicted'
      });
    }

    if (summary.aspartimideRisk > 0.12) {
      byproducts.push({
        name: '天冬酰亚胺副产物',
        mz: exactMassMH - 18.015,
        cause: '富含 Asp/Asn 的片段在重复脱保护过程中更容易暴露于碱促环化环境。'
      });
      massSpectrum.push({
        mz: exactMassMH - 18.015,
        abundance: Math.max(4, 28 - (purity * 0.14)),
        label: '[M-H2O+H]+',
        source: 'literature',
        reference: 'Domon & Aebersold (2006) Science 312:212-217'
      });
    }

    if (summary.cysCount >= 2 && !params.topology.includes('Disulfide') && !params.topology.includes('二硫键环化')) {
      byproducts.push({
        name: '半胱氨酸氧化二聚体',
        mz: (exactMassMH * 2) - 3.032,
        cause: '游离半胱氨酸在空气中可能被氧化，形成分子间二硫键二聚体。'
      });
    }

    if (summary.hydrophobicRatio > 0.48 && summary.length > 8) {
      byproducts.push({
        name: '聚集驱动的缺失副产物系列',
        mz: exactMassMH - 113.084,
        cause: '较长的疏水片段更容易引发树脂聚集，并抑制新鲜活化试剂在树脂中的扩散。'
      });
    }

    if (summary.difficultRatio > 0.24 && summary.reagentScore < 0.9) {
      optimizations.push('在当前 Fmoc SOP 下，建议将 HBTU/DIEA 提升为 DIC/Oxyma 或 HATU，并对富含 Arg/Cys/His 的片段采用双偶联。');
    }

    if (summary.bulkyRatio > 0.28) {
      optimizations.push('保持 3 × 5 分钟脱保护节奏不变，但对位阻较大的残基将偶联延长到 90-120 分钟，或额外重复一次偶联。');
    }

    if (summary.hydrophobicRatio > 0.42 && summary.solventScore < 0.8) {
      optimizations.push('对于疏水片段，应避免树脂过干，完整保留 DMF 洗涤流程，并考虑采用 DMF/DCM 辅助处理或分段双偶联。');
    }

    if (summary.aspartimideRisk > 0.12) {
      optimizations.push('对于 Asp/Asn 易形成天冬酰亚胺的序列，不要把碱处理时间延长到标准 3 × 5 分钟之外，并在偶联前彻底完成 DMF 洗涤。');
    }

    if (summary.cysCount >= 2 && !params.topology.includes('Disulfide') && !params.topology.includes('二硫键环化')) {
      optimizations.push('若存在多个游离半胱氨酸且并非二硫键环化目标，建议加强保护并在裂解或后处理阶段尽量采用惰性气氛。');
    }

    if ((params.topology.includes('Head-to-tail') || params.topology.includes('头尾环化')) && summary.length < 6) {
      optimizations.push('对于较短前体，头尾环化条件往往过于激进，建议优先优化线性前体构建后再尝试闭环。');
    }

    if (optimizations.length === 0) {
      optimizations.push('当前设置已经接近锁定的 Fmoc SOP 基线，建议继续严格执行 DMF 洗涤纪律，并对每个合并馏分做分析型 HPLC 和 LC-MS 验证。');
      optimizations.push('如果计划放大合成，请先保持相同的试剂当量、1 小时偶联窗口和 2 小时裂解条件，再考虑改变化学体系。');
    }

    massSpectrum.sort((left, right) => left.mz - right.mz);

    return {
      purity,
      yield: predictedYield,
      exactMass: exactMassMH,
      massSpectrum,
      byproducts: byproducts.slice(0, 4),
      optimizations: optimizations.slice(0, 3),
      literaturePeaks: [
        ...LITERATURE_PEAKS.adducts,
        ...LITERATURE_PEAKS.neutralLosses,
        ...LITERATURE_PEAKS.synthesisByproducts
      ]
    };
  }
}
