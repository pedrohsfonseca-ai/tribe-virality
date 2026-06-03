# CLAUDE.md — TRIBE v2 → NEURO CONTENT SCORE → Backtest

> Este arquivo é o contexto do projeto. Leia inteiro antes de agir. Está em pt-BR;
> termos técnicos em inglês. O dono do projeto (Pedro) é técnico, trabalha em
> pt-BR, e quer análise honesta — não hype. Se algo não tiver evidência, diga.

## 0. Objetivo em uma frase
Testar, com vídeos que o Pedro JÁ TEM, se a atividade cerebral prevista pelo
**TRIBE v2** (modelo open-source da Meta FAIR) tem poder preditivo sobre a
performance real do conteúdo dele — e, se tiver, empacotar como ferramenta de
triagem de criativo.

## 1. Enquadramento honesto (não viole isto)
- **O TRIBE prevê atividade cerebral (fMRI), NÃO viralidade.** A saída é uma matriz
  `[T, ~20484]` de ativação cortical na malha fsaverage5 do "sujeito médio".
  Qualquer "score de viralidade" é uma CAMADA DE INTERPRETAÇÃO por cima — só vira
  preditor depois de validado contra dados reais.
- **A métrica de sucesso é o BACKTEST**, não o número bonito: Spearman ρ entre score
  previsto e performance real + top-k hit rate. Sem isso, o score é hipótese. Não
  apresente o score como preditivo antes do backtest passar.
- **Viés cultural:** TRIBE foi treinado em fMRI majoritariamente não-brasileiro. O
  "sujeito médio" dele ≠ público do Pedro. Por isso a calibração contra os dados
  DELE é o que importa.
- **Licença CC-BY-NC-4.0 (não-comercial).** Protótipo/pesquisa: ok. Operacionalizar
  como produto/decisão comercial: avisar que precisa de aval jurídico antes.
- Não invente acurácia. Com n < ~15 vídeos, reporte correlação mas deixe claro que
  é frágil. Não faça o classificador "decorar" os mesmos vídeos que avalia.

## 2. Onde isto roda
Pod **RunPod** (Ubuntu, template PyTorch 2.x), GPU **A40 48GB** (ou A100 40GB+).
O `predict()` do TRIBE tem pico de **~24–28 GB VRAM**. Network Volume em `/workspace`
guarda os pesos (`HF_HOME`, `TRIBE_CACHE`) pra não re-baixar a cada restart.

## 3. Arquivos do repositório
| Arquivo | Papel |
|---|---|
| `CLAUDE.md` | Este brief. |
| `setup.sh` | Provisiona o pod (deps de sistema, numpy<2.1, tribev2, nilearn, Claude Code). NÃO reinstala torch. |
| `smoke_test.py` | Valida a máquina inteira num clipe sintético ANTES dos vídeos reais. Tem que imprimir "MÁQUINA PRONTA". |
| `tribe_virality.py` | Camada de análise pura (sem GPU): vértice→rede (Destrieux), features neurais, score heurístico, backtest (Spearman/top-k/per-feature), calibração leave-one-out. JÁ TESTADA com dados sintéticos. |
| `tribe_virality_mvp.ipynb` | Versão notebook (Colab) do mesmo fluxo. Redundante com os scripts; use os scripts no pod. |
| `metrics_template.csv` | Modelo da planilha de performance real (1 linha por vídeo, `video_id` + métricas). |

## 4. API REAL do TRIBE (confirmada no repo facebookresearch/tribev2 — não alucine)
```python
from tribev2.demo_utils import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=os.environ["TRIBE_CACHE"])
df = model.get_events_dataframe(video_path="clip.mp4")   # ou text_path=/audio_path=
preds, segments = model.predict(events=df)               # preds: np.ndarray [T, ~20484]
# preds está deslocado ~5s no passado (lag hemodinâmico) — ok p/ análise relativa.
# Acesso de baixo nível (se precisar das features dos encoders, não da predição cortical):
#   feats = model.extract_features(df)  # {modality: np.ndarray [n_layers, dim, T]}
```
- Texto usa **LLaMA 3.2-3B (gated)** → precisa de `huggingface-cli login` + aceitar a licença.
- Exige **numpy < 2.1**.

## 5. Pipeline (fluxo lógico)
1. Para cada vídeo: `get_events_dataframe` → `predict` → `preds [T, V]`.
2. `tribe_virality.roi_timeseries(preds, vertex_networks)` → curva por rede funcional
   (visual, auditory, language, social_tpj, attention, salience).
3. `tribe_virality.neuro_features(roi)` → `hook_power`, `retention_power`,
   `engagement_trigger` + médias/picos por rede.
4. `tribe_virality.score_videos(features_df)` → `neuro_score` 0–10 (pesos PRIOR).
5. Juntar com `metrics_template.csv` preenchido pelo Pedro (a métrica-alvo vai em
   `ACTUAL_COL`, ex.: `reach_vs_typical`, `retention_pct`, `shares`).
6. `tribe_virality.backtest(...)` → Spearman ρ, top-k hit rate, correlação por feature.
7. `tribe_virality.calibrate_weights(...)` (só com n ≥ ~15) → refit leave-one-out.

## 6. ⚠️ PONTO FRÁGIL #1 — alinhamento vértice→atlas
O `tribe_virality.py` e o `smoke_test.py` montam o atlas como
`np.concatenate([map_left, map_right])` (LH 10242 + RH 10242 = 20484), assumindo a
ordem padrão fsaverage5. **Se `preds.shape[1] != 20484`** (o TRIBE pode mascarar a
*medial wall* e devolver menos vértices), o mapeamento rede↔vértice fica errado e
TODO o resto vira lixo silencioso. Antes de confiar em qualquer resultado:
- Confirme `preds.shape[1]` contra o `tribe_demo.ipynb` oficial e o `utils_fmri.py`
  do repo (que tem "Surface projection / ROI analysis").
- Se houver máscara de medial wall, descubra o índice dos vértices válidos e
  alinhe o atlas a ele (subset dos 20484), em vez de assumir o concat cru.
Esta é a primeira coisa a verificar/consertar.

## 7. Passo a passo (o que fazer, em ordem)
1. `bash setup.sh`
2. `huggingface-cli login` (token read; aceitar licença do `meta-llama/Llama-3.2-3B`)
3. `python smoke_test.py` → corrigir o que falhar até sair "MÁQUINA PRONTA"
   (provável ajuste: o alinhamento do item 6).
4. Receber os `.mp4` do Pedro em `/workspace/project/videos/` (id = nome do arquivo).
5. Rodar o loop de inferência → `tribe_features.csv`.
6. Pedro preenche o `metrics_template.csv` com performance real → `metrics.csv`.
7. Rodar o backtest. Reportar: ρ, top-k hit rate, qual feature mais correlaciona,
   e a interpretação honesta (≥0.6 promissor / 0.3–0.6 sinal fraco / <0.3 não serve
   como preditor).
8. (Opcional, paridade com o reel original) camada LLM com a Anthropic SDK pra
   gerar a EXPLICAÇÃO de cada vídeo a partir das features — o número quem valida é
   o backtest, não o LLM.

## 8. Convenções / o que NÃO fazer
- NÃO reinstalar torch (vem com CUDA no template; quebra a GPU).
- Manter `numpy < 2.1`.
- NÃO apresentar `neuro_score` como "vai viralizar". É triagem relativa até o ρ provar.
- Persistir pesos no Network Volume (`HF_HOME`, `TRIBE_CACHE` em `/workspace`).
- Falhar cedo e com mensagem clara (padrão do `smoke_test.py`).
- Custos: A40 ~US$0,35/h; um MVP de ~15 vídeos fica em ~1–2h de GPU. Desligue o pod
  ao terminar (o Network Volume continua cobrando ~US$0,07/GB/mês mesmo parado).

## 9. Primeira ação sugerida
Rode o item 7.1–7.3. Se o `smoke_test.py` acusar mismatch de vértices, conserte o
alinhamento (item 6) antes de qualquer outra coisa, e só então peça os vídeos.
