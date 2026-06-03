# tribe-virality

Pipeline para testar se a atividade cerebral prevista pelo **TRIBE v2** (Meta FAIR)
tem poder preditivo sobre a performance real de reels — e, se tiver, virar uma
ferramenta de triagem de criativo. Ver [CLAUDE.md](CLAUDE.md) para o brief completo
e o enquadramento honesto.

## Fluxo

```
setup.sh            # 1. provisiona o pod (deps, numpy<2.1, tribev2, nilearn)
hf auth login           # 2. token 'read' + aceitar licença do Llama-3.2-3B
python smoke_test.py    # 3. valida a máquina -> "MÁQUINA PRONTA"
                        #    (descobre o nº REAL de vértices — ponto frágil #1)
# 4. colocar os .mp4 em $PROJECT_DIR/videos/  (id = nome do arquivo)
python run_inference.py # 5. roda os vídeos -> tribe_features.csv
# 6. preencher metrics_template.csv -> metrics.csv (performance real)
python run_backtest.py retention_pct   # 7. score previsto vs real (Spearman/top-k)
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `tribe_virality.py` | Núcleo de análise (sem GPU): vértice→rede, features, score, backtest, calibração. |
| `test_local.py` | Testa o núcleo offline com dados sintéticos. |
| `setup.sh` | Provisiona o pod. NÃO reinstala torch. |
| `smoke_test.py` | Valida a máquina inteira num clipe sintético. |
| `run_inference.py` | Loop de inferência → `tribe_features.csv`. |
| `run_backtest.py` | Backtest contra `metrics.csv`. |
| `metrics_template.csv` | Modelo da planilha de performance real. |

## Onde roda
RunPod, GPU A40 48GB (`predict()` tem pico ~24–28 GB VRAM). Pesos persistem no
Network Volume (`/workspace`) via `HF_HOME` e `TRIBE_CACHE`.

> ⚠️ TRIBE prevê atividade cerebral, **não** viralidade. O "score" é interpretação;
> só vira preditor depois que o backtest provar correlação contra dados reais.
> Licença CC-BY-NC-4.0 (não-comercial) — protótipo/pesquisa ok.
