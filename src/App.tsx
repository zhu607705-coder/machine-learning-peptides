import { useEffect, useState, type FormEvent } from 'react';
import { LocalPeptideModel } from './lib/PeptideModel';
import { standardFmocSopProfile } from './lib/fmocSop';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import {
  Settings2,
  Activity,
  Beaker,
  Zap,
  AlertTriangle,
  Lightbulb,
  Loader2,
  Shield
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface MassPeak {
  mz: number;
  abundance: number;
  label: string;
  source: 'predicted' | 'literature';
  reference?: string;
}

interface LiteraturePeak {
  mz: number;
  label: string;
  description: string;
  condition: string;
  reference: string;
}

interface PredictionResult {
  purity: number;
  yield: number;
  exactMass: number;
  massSpectrum: MassPeak[];
  byproducts: { name: string; mz: number; cause: string }[];
  optimizations: string[];
  literaturePeaks: LiteraturePeak[];
}

const LOADING_MESSAGES = [
  '正在解析多肽序列拓扑结构...',
  '正在检索最新合成文献与实验数据...',
  '正在计算精确分子质量与同位素分布...',
  '正在推演潜在副反应与杂质谱...',
  '正在生成质谱图与优化策略...'
];

function ReadonlyField({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-mono text-gray-400 uppercase tracking-wider">{label}</label>
      <div className="w-full rounded-lg border border-white/10 bg-white/[0.03] px-4 py-2.5 text-sm text-gray-200">
        {value}
      </div>
    </div>
  );
}

export default function App() {
  const diagnostics = LocalPeptideModel.getModelDiagnostics();
  const sop = standardFmocSopProfile;

  const [sequence, setSequence] = useState(sop.defaults.sequence);
  const [topology, setTopology] = useState(sop.defaults.topology);
  const [couplingReagent, setCouplingReagent] = useState(sop.defaults.couplingReagent);

  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingMessageIndex, setLoadingMessageIndex] = useState(0);

  useEffect(() => {
    if (!isPredicting) {
      setLoadingMessageIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setLoadingMessageIndex((current) => (current + 1) % LOADING_MESSAGES.length);
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [isPredicting]);

  const handlePredict = async (e: FormEvent) => {
    e.preventDefault();
    setIsPredicting(true);
    setError(null);
    setLoadingMessageIndex(0);

    try {
      const data = await LocalPeptideModel.predict({
        sequence,
        topology,
        couplingReagent,
        solvent: sop.defaults.solvent,
        temperature: sop.defaults.temperature,
        cleavageTime: sop.defaults.cleavageTime
      });

      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-300 font-sans selection:bg-emerald-500/30">
      <header className="sticky top-0 z-10 border-b border-white/10 bg-[#111] px-6 py-4 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-emerald-500/50 bg-emerald-500/20">
              <Activity className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-lg font-medium tracking-tight text-white">DeepPeptide 预测器</h1>
              <p className="text-xs font-mono uppercase tracking-wider text-gray-500">
                v2.5.0 // Fmoc 标准规程已锁定
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-sm font-mono text-gray-500">
            <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-emerald-300">
              {sop.processLabel}
            </span>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              模型已准备就绪
            </div>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-8 px-6 py-8 lg:grid-cols-12">
        <div className="space-y-6 lg:col-span-4">
          <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 p-5 shadow-xl">
            <div className="mb-3 flex items-center gap-2">
              <Shield className="h-5 w-5 text-amber-300" />
              <h2 className="text-sm font-semibold uppercase tracking-wider text-white">核心安全</h2>
            </div>
            <p className="text-sm leading-relaxed text-amber-100/85">{sop.safetyNotice}</p>
          </div>

          <div className="rounded-xl border border-white/10 bg-[#111] p-6 shadow-xl">
            <div className="mb-6 flex items-center gap-2 border-b border-white/5 pb-4">
              <Settings2 className="h-5 w-5 text-gray-400" />
              <h2 className="text-sm font-semibold uppercase tracking-wider text-white">
                Fmoc SPPS 标准参数
              </h2>
            </div>

            <div className="mb-5 grid gap-4 md:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              <div className="rounded-lg border border-emerald-500/15 bg-emerald-500/5 p-4">
                <div className="flex items-center justify-between gap-3 text-[11px] font-mono uppercase tracking-wider text-emerald-300">
                  <span>调优合成基准</span>
                  <span>训练/验证/测试</span>
                </div>
                <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <p className="text-gray-500">划分</p>
                    <p className="font-mono text-white">
                      {diagnostics.splitSizes.train}/{diagnostics.splitSizes.validation}/{diagnostics.splitSizes.test}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500">验证均方根误差</p>
                    <p className="font-mono text-white">{diagnostics.validation.combinedRmse.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-gray-500">测试均方根误差</p>
                    <p className="font-mono text-white">{diagnostics.test.combinedRmse.toFixed(2)}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-lg border border-cyan-500/15 bg-cyan-500/5 p-4">
                <div className="text-[11px] font-mono uppercase tracking-wider text-cyan-300">
                  锁定实验流程
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-gray-500">Fmoc 脱除</p>
                    <p className="font-mono text-white">3 × 5 分钟</p>
                  </div>
                  <div>
                    <p className="text-gray-500">偶联</p>
                    <p className="font-mono text-white">室温 1 小时</p>
                  </div>
                  <div>
                    <p className="text-gray-500">DMF 洗涤</p>
                    <p className="font-mono text-white">5-6 次临界步骤</p>
                  </div>
                  <div>
                    <p className="text-gray-500">裂解</p>
                    <p className="font-mono text-white">2 小时 TFA 混合液</p>
                  </div>
                </div>
              </div>
            </div>

            <form onSubmit={handlePredict} className="space-y-5">
              <div className="space-y-2">
                <label className="text-xs font-mono uppercase tracking-wider text-gray-400">序列</label>
                <input
                  type="text"
                  value={sequence}
                  onChange={(e) => setSequence(e.target.value)}
                  className="w-full rounded-lg border border-white/10 bg-[#0a0a0a] px-4 py-2.5 font-mono text-sm text-white transition-all focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
                  placeholder="例如：H-Gly-Ala-Val-Leu-Ile-OH"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-mono uppercase tracking-wider text-gray-400">拓扑</label>
                <select
                  value={topology}
                  onChange={(e) => setTopology(e.target.value)}
                  className="w-full appearance-none rounded-lg border border-white/10 bg-[#0a0a0a] px-4 py-2.5 text-sm text-white transition-all focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
                >
                  <option>线性</option>
                  <option>头尾环化</option>
                  <option>二硫键环化</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-xs font-mono uppercase tracking-wider text-gray-400">偶联试剂</label>
                <select
                  value={couplingReagent}
                  onChange={(e) => setCouplingReagent(e.target.value)}
                  className="w-full appearance-none rounded-lg border border-white/10 bg-[#0a0a0a] px-4 py-2.5 text-sm text-white transition-all focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/50"
                >
                  <option>HBTU/DIEA</option>
                  <option>DIC/Oxyma</option>
                  <option>HATU</option>
                </select>
              </div>

              <ReadonlyField label="溶剂体系" value={sop.defaults.solvent} />
              <ReadonlyField label="反应温度" value={sop.defaults.temperature} />
              <ReadonlyField label="裂解时间" value={sop.defaults.cleavageTime} />

              <button
                type="submit"
                disabled={isPredicting}
                className="mt-6 flex w-full items-center justify-center gap-2 rounded-lg bg-white py-3 font-medium text-black transition-colors hover:bg-gray-200 disabled:bg-white/20 disabled:text-white/50"
              >
                {isPredicting ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    正在执行推断...
                  </>
                ) : (
                  <>
                    <Zap className="h-5 w-5" />
                    运行预测模型
                  </>
                )}
              </button>
            </form>
          </div>

          <div className="rounded-xl border border-white/10 bg-[#111] p-6">
            <div className="mb-4 flex items-center gap-2 border-b border-white/5 pb-3">
              <Beaker className="h-5 w-5 text-emerald-400" />
              <h3 className="text-sm font-semibold uppercase tracking-wider text-white">关键控制点</h3>
            </div>
            <div className="grid gap-3">
              {sop.criticalControls.map((item) => (
                <div key={item.label} className="flex items-start justify-between gap-3 rounded-lg border border-white/5 bg-white/[0.03] px-4 py-3">
                  <span className="text-xs font-mono uppercase tracking-wider text-gray-400">{item.label}</span>
                  <span className="text-right text-sm text-white">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-6 lg:col-span-8">
          {error && (
            <div className="flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-red-400">
              <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {!result && !isPredicting && !error && (
            <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
              <div className="rounded-xl border border-white/10 bg-[#111] p-6">
                <div className="mb-4 flex items-center gap-2 border-b border-white/5 pb-3">
                  <Activity className="h-5 w-5 text-emerald-400" />
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-white">标准操作程序工作流程图</h3>
                </div>
                <div className="space-y-4">
                  {sop.steps.map((step, index) => (
                    <div key={step.title} className="flex gap-4">
                      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-emerald-500/20 bg-emerald-500/10 text-xs font-mono text-emerald-400">
                        {index + 1}
                      </div>
                      <div>
                        <p className="text-sm font-medium text-white">{step.title}</p>
                        <p className="mt-1 text-sm leading-relaxed text-gray-400">{step.detail}</p>
                        {step.emphasis ? (
                          <p className="mt-1 text-xs font-mono uppercase tracking-wider text-amber-300">{step.emphasis}</p>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-6">
                <div className="rounded-xl border border-white/10 bg-[#111] p-6">
                  <div className="mb-4 flex items-center gap-2 border-b border-white/5 pb-3">
                    <Shield className="h-5 w-5 text-cyan-400" />
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-white">纯化终点</h3>
                  </div>
                  <div className="space-y-3">
                    {sop.purification.map((item) => (
                      <div key={item.label} className="rounded-lg border border-white/5 bg-white/[0.03] px-4 py-3">
                        <p className="text-xs font-mono uppercase tracking-wider text-gray-500">{item.label}</p>
                        <p className="mt-1 text-sm text-white">{item.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex min-h-[180px] flex-col items-center justify-center rounded-xl border border-dashed border-white/10 bg-[#111] text-gray-500">
                  <Beaker className="mb-4 h-12 w-12 opacity-20" />
                  <p className="text-sm font-mono tracking-wide">等待输入 Fmoc 标准规程参数</p>
                  <p className="mt-2 max-w-sm text-center text-xs leading-relaxed text-gray-500">
                    输入目标多肽序列，在锁定的 Fmoc 工艺基线上运行预测，即可查看纯度、收率和副产物风险。
                  </p>
                </div>
              </div>
            </div>
          )}

          {isPredicting && (
            <div className="overflow-hidden rounded-xl border border-emerald-500/10 bg-[#111]">
              <div className="grid min-h-[420px] grid-cols-1 lg:grid-cols-[0.95fr_1.05fr]">
                <div className="flex flex-col justify-center border-b border-white/5 bg-[radial-gradient(circle_at_top,#10b98126,transparent_55%)] px-8 py-10 lg:border-b-0 lg:border-r">
                  <div className="mb-6 flex items-center gap-3">
                    <div className="relative">
                      <span className="absolute inset-0 rounded-full bg-emerald-400/30 blur-xl animate-pulse" />
                      <span className="relative flex h-16 w-16 items-center justify-center rounded-full border border-emerald-400/40 bg-emerald-400/10">
                        <span className="h-5 w-5 rounded-full bg-emerald-400 animate-pulse" />
                      </span>
                    </div>
                    <div>
                      <p className="text-sm font-mono uppercase tracking-[0.35em] text-emerald-300">模型运行中</p>
                      <p className="mt-2 text-sm leading-relaxed text-gray-400">
                        正在基于 Fmoc 标准规程联动序列特征、质谱规则和本地模型输出生成分析结果。
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-emerald-500/15 bg-black/20 p-5">
                    <p className="text-xs font-mono uppercase tracking-[0.3em] text-gray-500">当前进度</p>
                    <p className="mt-4 text-xl font-medium leading-relaxed text-white">
                      {LOADING_MESSAGES[loadingMessageIndex]}
                    </p>
                  </div>
                </div>

                <div className="flex flex-col justify-center px-8 py-10">
                  <div className="mb-5 flex items-center gap-2">
                    <Loader2 className="h-5 w-5 animate-spin text-emerald-400" />
                    <p className="text-sm font-semibold uppercase tracking-wider text-white">进度轮播</p>
                  </div>
                  <div className="space-y-3">
                    {LOADING_MESSAGES.map((message, index) => {
                      const active = index === loadingMessageIndex;
                      return (
                        <div
                          key={message}
                          className={cn(
                            'rounded-xl border px-4 py-3 transition-all duration-500',
                            active
                              ? 'border-emerald-400/30 bg-emerald-500/10 shadow-[0_0_40px_rgba(16,185,129,0.12)]'
                              : 'border-white/5 bg-white/[0.03]'
                          )}
                        >
                          <div className="flex items-start gap-3">
                            <div
                              className={cn(
                                'mt-1 h-2.5 w-2.5 shrink-0 rounded-full transition-all duration-500',
                                active ? 'bg-emerald-400 shadow-[0_0_14px_rgba(16,185,129,0.9)] animate-pulse' : 'bg-white/20'
                              )}
                            />
                            <p className={cn('text-sm leading-relaxed', active ? 'text-white' : 'text-gray-500')}>
                              {message}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}

          {result && !isPredicting && (
            <div className="animate-in slide-in-from-bottom-4 space-y-6 fade-in duration-500">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-[#111] p-5">
                  <div className="absolute right-0 top-0 p-4 opacity-10 transition-opacity group-hover:opacity-20">
                    <Activity className="h-16 w-16" />
                  </div>
                  <p className="mb-1 text-xs font-mono uppercase tracking-wider text-gray-400">粗品纯度估计</p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-light text-white">{result.purity.toFixed(1)}</span>
                    <span className="text-sm text-gray-500">%</span>
                  </div>
                  <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-white/5">
                    <div
                      className={cn('h-full rounded-full', result.purity > 80 ? 'bg-emerald-500' : result.purity > 60 ? 'bg-yellow-500' : 'bg-red-500')}
                      style={{ width: `${result.purity}%` }}
                    />
                  </div>
                </div>

                <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-[#111] p-5">
                  <div className="absolute right-0 top-0 p-4 opacity-10 transition-opacity group-hover:opacity-20">
                    <Beaker className="h-16 w-16" />
                  </div>
                  <p className="mb-1 text-xs font-mono uppercase tracking-wider text-gray-400">理论收率估计</p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-light text-white">{result.yield.toFixed(1)}</span>
                    <span className="text-sm text-gray-500">%</span>
                  </div>
                  <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-white/5">
                    <div
                      className={cn('h-full rounded-full', result.yield > 70 ? 'bg-emerald-500' : result.yield > 40 ? 'bg-yellow-500' : 'bg-red-500')}
                      style={{ width: `${result.yield}%` }}
                    />
                  </div>
                </div>

                <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-[#111] p-5">
                  <div className="absolute right-0 top-0 p-4 opacity-10 transition-opacity group-hover:opacity-20">
                    <Zap className="h-16 w-16" />
                  </div>
                  <p className="mb-1 text-xs font-mono uppercase tracking-wider text-gray-400">理论精确质量 [M+H]+</p>
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-light text-white">{result.exactMass.toFixed(2)}</span>
                    <span className="text-sm text-gray-500">Da</span>
                  </div>
                  <div className="mt-3 text-xs font-mono text-emerald-400/80">目标分子质量已完成校验</div>
                </div>
              </div>

              <div className="rounded-xl border border-white/10 bg-[#111] p-6">
                <div className="mb-4 flex items-center justify-between">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-white">预测质谱分布图</h3>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                      <div className="h-3 w-3 rounded-sm bg-emerald-500" />
                      <span className="text-xs text-gray-400">模型预测</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="h-3 w-3 rounded-sm bg-blue-500" />
                      <span className="text-xs text-gray-400">文献参考</span>
                    </div>
                  </div>
                </div>
                <div className="h-[300px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={result.massSpectrum} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                      <CartesianGrid stroke="#ffffff10" strokeDasharray="3 3" vertical={false} />
                      <XAxis
                        dataKey="mz"
                        stroke="#ffffff50"
                        tick={{ fill: '#ffffff50', fontSize: 12, fontFamily: 'monospace' }}
                        tickFormatter={(val) => val.toFixed(1)}
                        label={{ value: 'm/z', position: 'insideBottom', offset: -10, fill: '#ffffff50', fontSize: 12 }}
                      />
                      <YAxis
                        stroke="#ffffff50"
                        tick={{ fill: '#ffffff50', fontSize: 12, fontFamily: 'monospace' }}
                        label={{ value: 'Abundance (%)', angle: -90, position: 'insideLeft', fill: '#ffffff50', fontSize: 12 }}
                      />
                      <Tooltip
                        cursor={{ fill: '#ffffff05' }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload as MassPeak;
                            return (
                              <div className="max-w-xs rounded-lg border border-white/10 bg-[#1a1a1a] p-3 shadow-xl">
                                <div className="mb-2 flex items-center gap-2">
                                  <div className={cn('h-2 w-2 rounded-full', data.source === 'predicted' ? 'bg-emerald-500' : 'bg-blue-500')} />
                                  <span className="text-xs uppercase text-gray-400">{data.source === 'predicted' ? '模型推演' : '文献参考'}</span>
                                </div>
                                <p className="mb-1 font-mono text-sm text-white">{data.label}</p>
                                <p className="text-xs font-mono text-gray-400">
                                  m/z: <span className="text-emerald-400">{data.mz.toFixed(2)}</span>
                                </p>
                                <p className="text-xs font-mono text-gray-400">
                                  丰度: <span className="text-white">{data.abundance.toFixed(1)}%</span>
                                </p>
                                {data.reference ? <p className="mt-2 text-[10px] italic text-gray-500">{data.reference}</p> : null}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar
                        dataKey="abundance"
                        radius={[2, 2, 0, 0]}
                        shape={(props: any) => {
                          const { x, y, width, height, payload } = props;
                          const fill = payload.source === 'predicted' ? '#10b981' : '#3b82f6';
                          return <rect x={x} y={y} width={width} height={height} fill={fill} rx={2} ry={2} />;
                        }}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="mt-3 text-xs text-gray-500">
                  <span className="text-emerald-400">●</span> 神经网络推演峰
                  <span className="mx-2">|</span>
                  <span className="text-blue-400">●</span> 文献参考峰（b/y 碎片离子、亚胺离子）
                </p>
              </div>

              {result.literaturePeaks.length > 0 && (
                <div className="rounded-xl border border-white/10 bg-[#111] p-6">
                  <div className="mb-4 flex items-center gap-2 border-b border-white/5 pb-3">
                    <Beaker className="h-5 w-5 text-blue-400" />
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-white">文献峰位与碎片参考</h3>
                  </div>
                  <div className="max-h-[300px] space-y-3 overflow-y-auto">
                    {result.literaturePeaks.map((peak, idx) => (
                      <div key={idx} className="flex items-start gap-3 rounded-lg border border-white/5 bg-white/5 p-3">
                        <div className="w-16 shrink-0 text-right">
                          <span className="text-xs font-mono text-blue-400">
                            {peak.mz > 0 ? '+' : ''}
                            {peak.mz.toFixed(3)}
                          </span>
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-sm font-medium text-white">{peak.label}</p>
                          <p className="text-xs text-gray-400">{peak.description}</p>
                          <p className="mt-1 text-xs text-gray-500">条件：{peak.condition}</p>
                          <p className="mt-1 text-[10px] italic text-gray-600">{peak.reference}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <div className="rounded-xl border border-white/10 bg-[#111] p-6">
                  <div className="mb-6 flex items-center gap-2 border-b border-white/5 pb-4">
                    <AlertTriangle className="h-5 w-5 text-yellow-500" />
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-white">潜在副产物分析</h3>
                  </div>
                  <div className="space-y-4">
                    {result.byproducts.map((bp, idx) => (
                      <div key={idx} className="rounded-lg border border-white/5 bg-white/5 p-4 transition-colors hover:border-white/10">
                        <div className="mb-2 flex items-start justify-between gap-3">
                          <h4 className="text-sm font-medium text-white">{bp.name}</h4>
                          <span className="rounded bg-yellow-500/10 px-2 py-0.5 text-xs font-mono text-yellow-500">
                            m/z {bp.mz.toFixed(1)}
                          </span>
                        </div>
                        <p className="text-xs leading-relaxed text-gray-400">{bp.cause}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-xl border border-white/10 bg-[#111] p-6">
                  <div className="mb-6 flex items-center gap-2 border-b border-white/5 pb-4">
                    <Lightbulb className="h-5 w-5 text-emerald-400" />
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-white">工艺优化建议</h3>
                  </div>
                  <div className="space-y-4">
                    {result.optimizations.map((opt, idx) => (
                      <div key={idx} className="flex gap-3">
                        <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-emerald-500/20 bg-emerald-500/10">
                          <span className="text-xs font-mono text-emerald-400">{idx + 1}</span>
                        </div>
                        <p className="text-sm leading-relaxed text-gray-300">{opt}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
