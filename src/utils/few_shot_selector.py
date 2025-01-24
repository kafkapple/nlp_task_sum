import pandas as pd

class FewShotSelector:
    @staticmethod
    def select_examples(train_df, config):
        """few-shot 예제 선택"""
        method = config.custom_config.few_shot_selection
        n_samples = config.custom_config.n_few_shot_samples
        
        if method == "random":
            samples = train_df.sample(n_samples, random_state=config.general.seed)
        
        elif method == "max_len_summary":
            samples = train_df.nlargest(n_samples, "summary_len")
            
        elif method == "min_max_len_summary":
            min_samples = train_df.nsmallest(n_samples // 2, "summary_len")
            max_samples = train_df.nlargest(n_samples // 2, "summary_len")
            samples = pd.concat([min_samples, max_samples])
            
        elif method == "similarity":
            # 입력 대화와 유사한 예제 선택
            samples = FewShotSelector._select_by_similarity(train_df, n_samples)
            
        elif method == "diverse":
            # 다양한 주제/길이의 예제 선택
            samples = FewShotSelector._select_diverse_samples(train_df, n_samples)
            
        else:
            raise ValueError(f"Unknown selection method: {method}")
            
        return {
            "dialogue": samples['dialogue'].tolist(),
            "summary": samples['summary'].tolist()
        }
    
    @staticmethod
    def _select_by_similarity(df, n_samples):
        """유사도 기반 선택 로직"""
        pass
        
    @staticmethod
    def _select_diverse_samples(df, n_samples):
        """다양성 기반 선택 로직"""
        pass 