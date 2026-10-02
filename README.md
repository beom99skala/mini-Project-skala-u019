# DS Mini Project Day 2 — ESS 배터리 수명 예측

초기 100 cycle에서 관찰한 신호로 셀의 `cycle_life`를 예측하는 회귀 프로젝트입니다. Day 1 EDA에서 ΔQ(V)의 변화와 수명 사이에 강한 연관이 관찰되어, 이를 중심으로 피처를 설계했습니다. 분석 단위는 배터리 셀입니다.

## 데이터와 실행 방법

수업에서 제공한 MAT 파일을 내려받아 한 폴더에 둡니다. 원본 파일은 용량이 약 7.7 GB이므로 저장소에 포함하지 않습니다.

```text
data/
├── 2017-05-12_batch1.mat
├── 2018-02-20_batch2.mat
└── 2018-04-12_batch3.mat  # 선택적 외부 검증
```

Python 3.11 이상에서 다음을 실행합니다.

```bash
python -m pip install -r requirements.txt
python train.py --data-dir data
```

결과는 `results/`에 저장됩니다. `performance_reporting.csv`는 과제 지정 성능표, `metrics.json`은 상세 지표, `cv_comparison.csv`는 후보 비교, `quality_exclusions.csv`는 제외 대상과 사유, `batch1_holdout_predictions.csv`와 `batch2_predictions.csv`는 셀별 오차, `features_all_cells.csv`는 피처와 타깃 점검용입니다. 수업 노트북 `30-ESSHealth-scratch.ipynb`의 EDA 구조를 참고했으며, 실제 제공 파일명과 세 Batch의 MAT 구조에 맞는 추출 코드는 `train.py`에 구현했습니다.

## EDA에서 모델 설계로 연결한 내용

- 품질 확인 후 `cycle_life` 중앙값은 Batch 1이 772.5, Batch 2가 472, Batch 3이 964.5 cycle입니다. 이 제공 파일에서는 Batch 1과 Batch 2의 분포도 상당히 다릅니다. Batch 2에 단수명 셀이 몰려 있어 배치 간 일반화 검증이 핵심입니다.
- 초기 100 cycle의 정규화 Qd는 배치별 중앙값이 약 100%로 큰 열화가 드러나지 않았습니다. 후반 Qd 곡선에는 기울기 변화가 보이지만, 100 cycle 이후 정보는 예측 입력에 쓰지 않았습니다.
- Day 1 분석에서 ΔQ(V) 표준편차와 수명의 Spearman 상관은 Batch 1에서 약 -0.87이었습니다. 그래서 ΔQ 분산의 로그와 최솟값을 핵심 피처로 시험했습니다. 상관은 인과를 뜻하지 않습니다.
- 충전 속도와 수명의 관계는 배치마다 방향이 달라, 단일 C-rate를 수명의 직접 원인으로 가정하기 어렵습니다. 충전 정책은 이번 모델의 입력 대신 검증 분할의 그룹으로 사용했습니다.
- 초기 Qd·Qc 변동성 등에는 높은 피처 간 상관이 있어 입력 수를 제한했습니다. 추가 온도·충전시간 피처의 이득은 CV에서 별도로 확인했습니다.

## 피처와 데이터 처리

Cycle 10과 100의 보간 방전 곡선 차이 `ΔQ(V) = Qd100(V) - Qd10(V)`에서 분산의 로그, 최솟값, 절대 면적을 계산합니다. 전압별 면적은 MAT 파일의 실제 `Vdlin`을 사용합니다. 추가로 2–10 cycle 대비 95–100 cycle의 Qd 변화량, 2–100 cycle 평균 내부저항·최고온도·충전시간을 계산합니다. 모든 입력은 100 cycle 이내의 측정치에서만 얻습니다.

Batch 3의 곡선 시작 구간 차이를 고려해, 서로 다른 Batch의 원본 `Qdlin` 곡선을 직접 빼지 않습니다. 각 셀 안에서 Cycle 100과 10을 같은 전압축에 맞춰 비교한 뒤 요약값을 사용합니다. 그래도 배치 간 측정 조건 차이가 모두 사라진다고 볼 수는 없습니다.

두 피처 묶음을 Batch 1 학습 데이터에서 비교했습니다.

| 묶음 | 피처 |
|---|---|
| Core | `log10_delta_q_var`, `delta_q_min_ah`, `qd_change_ah`, `ir_mean_ohm` |
| Expanded | Core + `delta_q_abs_area_ahv`, `tmax_mean_c`, `charge_time_mean_min` |

제공된 `cycle_life`를 회귀 타깃으로 사용했습니다. 타깃이 없는 10개 셀은 제외했습니다. Batch 1의 10개 셀은 마지막 방전 용량이 0.885 Ah를 넘고 종료 기준 도달이 확인되지 않아 제외했습니다. 원 저자 코드는 이 중 일부를 나중 측정 파일과 연결하고 나머지를 제외하지만, 수업 제공 파일에는 연결에 필요한 2017-06-30 자료가 없습니다. Batch 3은 원 저자가 기록한 데이터 수집·잡음 문제에 해당하는 원본 위치 3·38·43·44번 셀을 추가로 제외했습니다. 모든 제외 상태와 마지막 방전 용량은 `results/features_all_cells.csv`에 기록했습니다. 마지막 용량은 품질 확인에만 사용하고 예측 피처에는 넣지 않았습니다. [원 저자 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)

피처가 결측이면 학습 fold의 중앙값으로 대체하도록 파이프라인을 구성했습니다. 이번 데이터에서는 선택된 피처에 결측이 없었습니다. 수업 노트북의 Qd 범위 필터는 시계열 행을 제거하므로 여기에는 적용하지 않았습니다.

## 모델 선택과 검증

Batch 1의 품질 확인을 통과한 36개 셀 중 29개를 학습, 7개를 hold-out으로 고정했습니다(`random_state=42`). 같은 충전 정책이 양쪽에 들어가지 않도록 그룹 단위로 분리했습니다. 학습 29개 안에서도 정책 그룹을 분리하는 15회 교차검증의 평균 MAPE로 피처 묶음과 하이퍼파라미터를 선택했습니다. 후보는 로그 타깃 Elastic Net, XGBoost, Random Forest입니다. 이 중 XGBoost는 Day 1에 제시한 비선형 비교 후보이며, Random Forest는 추가 비교용입니다. 중앙값 예측을 기준선으로 함께 보고합니다.

Elastic Net에서는 `alpha`와 `l1_ratio`, XGBoost에서는 `n_estimators`·`learning_rate`·`max_depth`·`min_child_weight`, Random Forest에서는 `n_estimators`·`max_features`·`max_depth`·`min_samples_leaf`를 학습 CV에서 비교했습니다. 데이터가 작아 트리 후보의 복잡도를 제한했습니다. XGBoost의 `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5`는 고정했습니다.

스케일링, 결측 대체, 하이퍼파라미터 선택은 각 학습 fold 안에서만 수행했습니다. Batch 1 hold-out 결과를 확인한 뒤, 선택된 설정으로 Batch 1 전체를 재학습해 Batch 2에서 최종 평가했습니다. Batch 2의 타깃은 Day 1 EDA에서 이미 살펴봤으므로 완전히 처음 보는 데이터라고 주장하지 않습니다. Day 2에서는 Batch 2 점수를 사용해 피처나 하이퍼파라미터를 다시 조정하지 않았습니다.

| 후보 | 피처 묶음 | 학습 CV MAPE |
|---|---|---:|
| Elastic Net, 로그 타깃 | Core | **10.34%** |
| Elastic Net, 로그 타깃 | Expanded | 10.54% |
| Random Forest | Core | 11.10% |
| Random Forest | Expanded | 11.20% |
| XGBoost | Core | 11.28% |
| XGBoost | Expanded | 11.77% |
| 중앙값 기준선 | Core | 18.46% |

선택된 설정은 Elastic Net의 `alpha=0.1`, `l1_ratio=0.2`입니다. 선형 모델이 두 트리 모델보다 Batch 1 CV에서 안정적이었습니다. 학습 CV는 작은 표본에서 수행되었으므로 단일 숫자를 일반화 성능으로 해석하지 않습니다.

## 성능 결과

과제 지정 형식에 맞춰 MAPE와 차이를 먼저 제시합니다. Gap은 뒤의 평가 구간 오차에서 앞의 평가 구간 오차를 뺀 값이며, 단위는 퍼센트포인트(pp)입니다.

| 구분 | MAPE / Gap | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 10.34% | 정책 그룹 분리, 15회 평균 |
| Valid (Batch 1 Hold-out) | 11.83% | 학습에 쓰지 않은 정책 그룹 7개 셀 |
| Test (Batch 2) | 31.18% | 최종 평가 39개 셀 |
| Gap (Train-Valid) | +1.49 pp | Valid − Train; 작은 표본에서 과적합 가능성 |
| Gap (Valid-Test) | +19.35 pp | Test − Valid; 배치 간 일반화 저하 |
| Gap (Target-Test) | +22.08 pp | Test − 과제에 제시된 논문 목표 9.1% |
| Test (Batch 3, 선택) | 12.52% | 품질 확인 후 40개 셀, 탐색적 평가 |
| Gap (Batch2-Batch3) | -18.66 pp | Batch 3 − Batch 2 |
| Gap (Target-Test, Batch 3) | +3.42 pp | Batch 3 − 과제 목표 9.1% |

추가 지표는 다음과 같습니다.

| 평가 구간 | 셀 수 | MAPE | MAE (cycle) | RMSE (cycle) | R² |
|---|---:|---:|---:|---:|---:|
| Batch 1 hold-out | 7 | **11.83%** | 101.59 | 114.20 | 0.640 |
| Batch 2 최종 테스트 | 39 | **31.18%** | 153.78 | 168.35 | 0.411 |
| Batch 3 선택적 외부 검증 | 40 | **12.52%** | 153.13 | 250.16 | 0.326 |
| Batch 2 중앙값 기준선 | 39 | 58.73% | 284.78 | 301.42 | -0.889 |

과제에 제시된 논문 목표 MAPE 9.1%보다 이번 Batch 2 결과가 높습니다. 원 저자는 2017-06-30 파일을 연결해 1차 테스트셋을 만들었고, 이번 과제는 2018-02-20 파일을 Batch 2로 지정합니다. 피처와 학습·테스트 구성이 다르므로 숫자를 직접 재현 결과로 해석할 수 없습니다. Batch 3의 `Qdlin` 시작 구간이 다른 문제도 완전히 보정했다고 주장할 수 없어, Batch 3 점수는 탐색적으로만 해석합니다.

## 오류 분석과 ESS 활용

Batch 2의 수명 중앙값은 472 cycle로 품질 확인 후 Batch 1의 772.5 cycle보다 짧습니다. Batch 2에서는 39개 중 35개 셀의 수명을 과대예측했습니다. 특히 실제 수명 500 cycle 미만인 28개 셀의 MAPE는 36.35%였습니다. 예를 들어 Batch 2의 19번 셀은 실제 449 cycle인데 754.3 cycle로 예측했습니다. 이는 Batch 1에서 학습한 관계가 Batch 2의 단수명 셀에 충분히 일반화되지 않음을 보여줍니다. 배치별 충전 프로토콜과 제조 조건의 차이도 가능한 설명이지만, 이번 분석만으로 원인을 확정할 수 없습니다. Batch 3 점수가 Batch 2보다 낮은 것은 Batch 3의 수명 분포가 학습 배치와 상대적으로 가깝기 때문일 수 있으며, 품질 제외와 곡선 정렬 문제도 있어 배치 일반화가 해결됐다는 뜻은 아닙니다.

ESS 운영에서 수명을 과대예측하면 교체·점검 시점을 늦출 위험이 있습니다. 따라서 이 모델은 Batch 1과 유사한 조건의 셀에서 참고 지표로만 사용하고, Batch 2와 같은 분포가 감지되면 별도 검증과 보수적인 점검 규칙이 필요합니다. 운영 적용 전에는 더 많은 셀과 배치에서 외부 검증하고, 충전 정책·온도 등 환경 차이를 기록해야 합니다.

## 한계

- Batch 1 학습 표본이 29개로 작고, hold-out도 7개뿐입니다.
- Batch 2는 Day 1 EDA에 사용되었습니다. 이번 모델 선택에는 Batch 1 학습 데이터만 사용했습니다.
- 이어 측정한 2017-06-30 파일이 없어 원 논문의 셀 연결 절차를 완전히 재현하지 않았습니다.
- Batch 3의 `Qdlin` 시작 구간 차이를 완전히 보정하지 못했으므로 선택 평가 결과는 참고용입니다.
- 이 저장소의 결과는 예측 실험이며 실제 ESS의 안전 또는 교체 결정을 자동화할 근거로 사용해서는 안 됩니다.

## 참고

- [Severson et al., Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8)
- 수업 제공 `30-ESSHealth-scratch.ipynb` 및 Batch 1–3 MAT 파일
- [DS Mini Project 과제 안내](https://actually-war-1ea.notion.site/DS-Mini-Project-32d7f4c8669380338a27f90c471c1fcb#32d7f4c86693806a834bf0aa7f712e12)
- [ML Hyperparameters 수업 자료](https://actually-war-1ea.notion.site/ML-Hyperparameters-2637f4c8669380379794c02e354d6527)

## 팀 구성

- 울산 1반 유희범: EDA, 피처 엔지니어링, 모델 비교, Batch 2·3 성능 평가
