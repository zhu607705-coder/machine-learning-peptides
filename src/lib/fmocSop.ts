export interface SopStep {
  title: string;
  detail: string;
  emphasis?: string;
}

export interface FmocSopProfile {
  name: string;
  version: string;
  processLabel: string;
  safetyNotice: string;
  defaults: {
    sequence: string;
    topology: string;
    couplingReagent: string;
    solvent: string;
    temperature: string;
    cleavageTime: string;
  };
  criticalControls: Array<{
    label: string;
    value: string;
  }>;
  steps: SopStep[];
  purification: Array<{
    label: string;
    value: string;
  }>;
}

export const standardFmocSopProfile: FmocSopProfile = {
  name: 'Fmoc 固相多肽合成',
  version: '标准规程已锁定',
  processLabel: '标准室温 Fmoc SPPS',
  safetyNotice:
    '所有 DMF、哌啶、DCM、乙醚和 TFA 相关操作必须在通风橱内完成，并全程佩戴护目镜、实验服和耐腐蚀丁腈手套。',
  defaults: {
    sequence: 'H-Gly-Ala-Val-Leu-Ile-OH',
    topology: '线性',
    couplingReagent: 'HBTU/DIEA',
    solvent: 'DMF',
    temperature: '室温',
    cleavageTime: '2 小时'
  },
  criticalControls: [
    { label: '树脂溶胀', value: 'DMF，30-60 分钟' },
    { label: 'Fmoc 脱保护', value: '20% 哌啶/DMF，3 × 5 分钟' },
    { label: '脱保护后洗涤', value: 'DMF × 5-6 次' },
    { label: '偶联窗口', value: '3-4 当量氨基酸，室温 1 小时' },
    { label: '切割体系', value: 'TFA/TIS/H2O = 95/2.5/2.5' },
    { label: '纯度门槛', value: '制备 HPLC + LC-MS，>95%' }
  ],
  steps: [
    {
      title: '树脂溶胀',
      detail: '装入起始树脂后，用 DMF 覆盖树脂表面 1-2 厘米，轻轻摇晃并溶胀 30-60 分钟。',
      emphasis: '排液后不要过度抽干树脂。'
    },
    {
      title: 'Fmoc 脱保护',
      detail: '使用新鲜 20% 哌啶/DMF 连续进行三次 5 分钟脱保护，每次之后都先快速 DMF 冲洗。',
      emphasis: '进入偶联前必须完成 5-6 次 DMF 彻底洗涤。'
    },
    {
      title: '氨基酸偶联',
      detail: '将 3-4 当量 Fmoc 氨基酸与偶联试剂预活化 3-5 分钟，然后在树脂上于室温摇晃反应 1 小时。',
      emphasis: '每次偶联后都要用 DMF 洗涤 3-4 次。'
    },
    {
      title: '末端收缩处理',
      detail: '最后一次脱保护后，用 DCM 置换 DMF 3-5 次，并在真空下将树脂干燥至自由流动的砂状状态。',
      emphasis: '这样可以实现更干净的 TFA 切割。'
    },
    {
      title: '裂解与分离',
      detail: '采用 TFA/TIS/H2O 95:2.5:2.5 切割 2 小时，随后进行氮吹、反相 HPLC 纯化、旋蒸和冻干。',
      emphasis: '只有经分析型 HPLC 和 LC-MS 确认合格的合并馏分才应进入冻干。'
    }
  ],
  purification: [
    { label: '上柱前过滤', value: '0.22 μm 或 0.45 μm 滤膜' },
    { label: '流动相 A', value: '超纯水 + 0.1% TFA' },
    { label: '流动相 B', value: '乙腈 + 0.1% TFA' },
    { label: '检测方式', value: '214 nm UV + LC-MS 确认' },
    { label: '冻干条件', value: '预冻后在低于 -50°C 冷阱下干燥 24-48 小时' }
  ]
};
