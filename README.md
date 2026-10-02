# ESS 배터리 수명 예측 — DS Mini Project Day 2

초기 100회 충·방전 데이터로 배터리 셀의 `cycle_life`를 예측한다. **Batch 1로 학습·검증하고 Batch 2로 최종 평가**했다. 과제 지표는 MAPE이며, Batch 3은 선택적 참고 평가다.

Day 1의 EDA 그래프·해석과 모델 설계 전략은 [설계 보고서 PDF](day1/DS-MINI-Design-EDA-Model-Strategy.pdf)에 있다.

## 프로젝트 개요 및 실행

- 데이터: MIT–Stanford Battery Dataset의 수업 제공 MAT 파일 (`2017-05-12_batch1.mat`, `2018-02-20_batch2.mat`, 선택적으로 `2018-04-12_batch3.mat`)
- 구조: `train.py`(데이터 처리·피처·모델·평가), `results/`(성능표·후보 비교·셀별 오차), `requirements.txt`
- 실행: Python 3.11+에서 `pip install -r requirements.txt` 후 `python train.py --data-dir data`
- 원본 MAT 파일은 크기가 약 7.7 GB여서 저장소에 포함하지 않았다. 파일을 `data/`에 넣으면 결과를 재생성할 수 있다.

## EDA → 피처 설계

- 품질 확인 후 Cycle Life 중앙값은 Batch 1 **772.5**, Batch 2 **472**, Batch 3 **964.5** cycle이다. 따라서 Batch 1에서의 좋은 점수가 다른 Batch로 이어지는지 확인해야 한다.
- Day 1 EDA에서 ΔQ(V) 분산과 수명의 Spearman 상관은 Batch 1에서 약 **−0.87**이었다. 단순 초기 용량보다 10→100회차 방전 곡선 변화가 유용할 것으로 보고 ΔQ(V) 요약값을 사용했다.
- 충전 정책별 수명 관계가 배치마다 달라 정책 자체는 입력에서 빼고, **정책 그룹이 학습·검증에 겹치지 않도록 분할**했다.
- 원논문을 참고해 ΔQ(V)의 평균·왜도·최솟값 변환, 방전용량의 2–100·91–100회차 기울기, 2회차 용량과 초기 최대 용량 차이를 추가 실험했다. 모두 100회차 이내 정보만 썼다.

## 모델링 전략 및 구현

품질 확인 후 Batch 1 **36개 셀** 중 29개를 학습, 7개를 hold-out으로 분리했다. 학습 29개에서 정책 그룹을 분리한 15회 교차검증의 평균 MAPE로 하이퍼파라미터를 선택했다. 후보는 로그 타깃 **Elastic Net**, Random Forest, XGBoost였다. 결측 대체와 Elastic Net 표준화는 학습 fold 안에서 수행했다. 주 결과는 해석이 쉬운 핵심 4개 피처 모델로 보고하고, 추가 피처는 비교 실험으로 남겼다.

| 피처 묶음·모델 | Batch 1 CV MAPE |
|---|---:|
| 핵심 피처 4개 + Elastic Net (주 결과) | 10.34% |
| 논문 참고 피처 11개 + Elastic Net (비교 실험) | **7.78%** |
| 용량 기울기만 추가 + Elastic Net | 10.39% |
| ΔQ 형태만 추가 + Elastic Net | 10.49% |
| 논문 참고 피처 11개 + Random Forest | 10.76% |
| 논문 참고 피처 11개 + XGBoost | 10.97% |

주 결과 모델은 **핵심 피처 4개를 쓰는 Elastic Net (`alpha=0.1`, `l1_ratio=0.2`)**이다. 타깃을 로그 변환해 학습한 후 cycle 단위로 역변환한다. 전체 후보와 탐색값은 [`results/cv_comparison.csv`](results/cv_comparison.csv), 실제 피처 계산은 [`train.py`](train.py)에 있다.

## 성능 결과

Gap은 뒤 구간 MAPE에서 앞 구간 MAPE를 뺀 값이며 단위는 **%p**다.

| 구분 | MAPE / Gap | 해석 |
|---|---:|---|
| Train (Batch 1 CV) | **10.34%** | 정책 그룹 분리, 15회 평균 |
| Valid (Batch 1 Hold-out) | **11.83%** | 미사용 정책 그룹 7개 셀 |
| Test (Batch 2) | **31.18%** | 평가 39개 셀 |
| Gap (Train-Valid) | +1.49%p | 과적합 여부 점검 |
| Gap (Valid-Test) | **+19.35%p** | 배치 간 일반화 저하 |
| Gap (Target-Test) | **+22.08%p** | 과제에 제시된 논문 목표 9.1% 대비 |
| Test (Batch 3, 선택) | 12.52% | 품질 확인 후 40개 셀, 참고용 |
| Gap (Batch 2–Batch 3) | −18.66%p | Batch 3 − Batch 2 |
| Gap (Target–Batch 3) | +3.42%p | Batch 3 − 9.1% |

논문 참고 피처를 추가하면서 Batch 1 CV는 **10.34→7.78%**, hold-out은 **11.83→9.20%**로 개선됐지만 Batch 2는 **31.18→40.77%**로 악화됐다. 그래서 추가 피처를 주 모델에 채택하지 않았다. **이 결정에는 이미 확인한 Batch 2 결과가 반영됐으므로, 위 Batch 2 점수는 독립적인 블라인드 최종평가로 해석할 수 없다.** 배치 간 일반화의 탐색적 비교 결과다. 논문의 데이터 구성·전처리도 이번 과제와 달라 9.1%는 참고 목표다.

## 오류 분석 및 ESS 도메인 해석

Batch 2는 39개 중 **35개 셀을 과대예측**했다. 특히 실제 수명 500회 미만인 28개의 MAPE가 약 **36.35%**였고, 19번 셀은 실제 **449회**를 **754.3회**로 예측했다. Batch 1보다 단수명 셀이 많은 Batch 2의 분포 차이가 주요 가설이다. 피처 중앙값도 Batch 1 대비 Batch 2에서 이동했으며, 특히 2–100회차 용량 기울기는 Batch 1 IQR의 **1.87배**, 91–100회차 기울기는 **2.29배** 차이였다([`results/feature_shift_batch1_batch2.csv`](results/feature_shift_batch1_batch2.csv)). 이는 추가 피처의 배치 민감성 가능성을 시사하지만 원인을 확정하지는 않는다.

ESS에서 수명 과대예측은 점검·교체 지연으로 이어질 수 있다. 따라서 이 모델을 실제 교체 결정을 자동화하는 데 사용해서는 안 된다. 더 많은 배치의 외부 검증, 환경·충전 조건 기록, 배치 차이 보정이 필요하다. Batch 3은 수명 분포와 `Qdlin` 시작 구간이 달라 점수를 탐색적으로만 해석했다.

## 데이터 처리·한계·참고

Batch 1에서 수명 종료가 관측되지 않은 10개, Batch 2·3의 타깃 결측 10개, 원 저자의 Batch 3 품질 제외 4개를 제거했다. 기준과 셀 번호는 [`results/quality_exclusions.csv`](results/quality_exclusions.csv)에 있다. 원 논문에서 일부 셀을 연결한 2017-06-30 파일은 수업 데이터에 없어 처리 절차를 완전히 재현하지 못했다. Batch 1 표본도 작아 성능의 불확실성이 크다.

- [Severson et al. (2019), *Nature Energy*](https://doi.org/10.1038/s41560-019-0356-8) · [원 저자 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m)
- 수업 제공 `30-ESSHealth-scratch.ipynb` 및 Batch 1–3 MAT 파일

## 팀 구성

- 울산 1반 유희범: EDA, 피처 엔지니어링, 모델 개발, Batch 2·3 평가
