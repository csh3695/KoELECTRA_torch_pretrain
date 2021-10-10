# KoELECTRA_torch_pretrain

- **Simplified** pytorch implementation for **KoELECTRA** additional-training.
    - **Pretrained KoELECTRA** 모델을 torch 기반으로 특정 text 필드(신문기사 등)의 데이터로 **추가학습**하기 위한 코드입니다.
    - 기존 KoELECTRA 학습 조건의 대부분을 그대로 사용합니다.
    - 모델 구조와 관련된 configuration 항목을 제거하여 단순화하였습니다.
    - Preprocessing은 신문 기사를 위한 전처리 로직의 구현입니다.

### Reference
[Pretrained ELECTRA Model for Korean](https://github.com/monologg/KoELECTRA)
