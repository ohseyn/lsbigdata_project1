!pip install pyreadstat
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_welfare = pd.read_spss("Data/Koweps_hpwc14_2019_beta2.sav")
welfare = raw_welfare.copy()
welfare.shape
welfare.info()
welfare.describe()

welfare = welfare.rename(
    columns = {"h14_g3" :"sex",
                "h14_g4" : "birth",
                "h14_g10" : "marriage_type",
                "h14_g11" : "religion",
                "p1402_8aq1" : "income",
                "h14_eco9" : "code_job",
                "h14_reg7" : "code_region"})
welfare = welfare[["sex", "birth", "marriage_type", "religion", "income", "code_job", "code_region"]]

welfare["sex"].dtypes # 변수 타입 출력
welfare["sex"].value_counts() # 이상치 확인
welfare["sex"].isna().sum() # 결측치 확인
welfare["sex"]=np.where(welfare["sex"]==1, "male", "female") # 성별 항목 이름 부여
sns.countplot(data = welfare, x = "sex") # 빈도 막대 그래프 그리기
plt.show()
plt.clf()

welfare["income"].describe() # 이상치 확인
welfare["income"].isna().sum() # 결측치 확인
welfare["income"]=np.where(welfare["income"] == 9999, np.nan, welfare["income"]) # 이상치 결측 처리

sex_income = welfare.dropna(subset = "income")\
            .groupby("sex", as_index=False)\
            .agg(mean_income=("income", "mean")) # 데이터 프레임
sns.barplot(data = sex_income, x = "sex", y ="mean_income") # x,y 값 알아야 작성 가능
plt.show()
plt.clf()

# 숙제: 위 그래프에서 각 성별 95% 신뢰구간 계산 후 그리기. 위 아래 검정색 막대기로 표시

welfare["birth"].describe() # 이상치 확인 
welfare["birth"].isna().sum() # 결측치 확인인
welfare["birth"]=np.where(welfare["birth"] == 9999, np.nan, welfare["birth"]) # 결측 처리
welfare = welfare.assign(age = 2019 - welfare["birth"]) # 나이 변수 만들기
welfare["age"].describe() # 통계량 구하기
sns.histplot(data = welfare, x = "age") # 히스토그램 그리기
plt.show()
plt.clf()

age_income = welfare.dropna(subset = "income")\ # income 결측치 제거
            .groupby("age", as_index=False)\ # age별 분리 
            .agg(mean_income=("income", "mean")) # income 평균 구하기
sns.lineplot(data = age_income, x = "age", y = "mean_income") # 선 그래프 그리기
plt.show()
plt.clf()

# 나이별 income 칼럼에서 na 개수 세기
# 나이별로 무응답 사람 수(일을 하지 않으니 무응답자 수가 많아짐)
my_df = welfare.assign(income_na = welfare["income"].isna())\
            .groupby("age", as_index=False)\
            .agg(n=("income_na", "sum"))
sns.barplot(data = my_df, x = "age", y = "n")
plt.show()
plt.clf()

welfare["age"]
welfare = welfare.assign(ageg = np.where(welfare["age"] < 30, "young", 
                                np.where(welfare["age"] <= 59, "middle", "old"))) # 연령대 변수 만들기
welfare["ageg"].value_counts() # 빈도 구하기
sns.countplot(data = welfare, x = "ageg")
plt.show()
plt.clf()

ageg_income = welfare.dropna(subset = "income")\
                    .groupby("ageg", as_index = False)\
                    .agg(mean_income = ("income", "mean")) 
sns.barplot(data = ageg_income, x = "ageg", y = "mean_income")
sns.barplot(data = ageg_income, x = "ageg", y = "mean_income", 
            order=["young", "middle", "old"]) 
plt.show()
plt.clf()

sex_income = welfare.dropna(subset = "income")\
                    .groupby(["ageg", "sex"], as_index = False)\
                    .agg(mean_income = ("income", "mean")) 
sns.barplot(data = sex_income, x = "ageg", y = "mean_income", 
						hue="sex", order=["young", "middle", "old"])
plt.show()
plt.clf()

sex_age =  welfare.dropna(subset = "income")\
                .groupby(["age", "sex"], as_index = False)\
                .agg(mean_income = ("income", "mean")) 
sns.lineplot(data = sex_age, x = "age", y = "mean_income", 
						hue="sex")
plt.show()
plt.clf()

#===============================================================
# cut 예문 -> 연령대별로 확인하기 위하여
vec_x = np.array([1, 7, 5, 4, 6, 3])
cut = pd.cut(vec_x, 3)
cut.describe()

# 한결언니
vec_x = np.random.randint(0, 100, 50)
age_max=119
bin_cut = [10 * i + 9 for i in np.arange(age_max//10 + 1)]
pd.cut(vec_x, bins = bin_cut)

# 다경언니
vec_x = np.random.randint(0, 100, 50)
bin_cut = np.array([0:110:10])
pd.cut(welfare["age"], bin_cut)

# 현주
welfare['ageg'] = pd.cut(welfare['age'],
                         bins=[0, 9, 19, 29, 39, 49, 59, 69, np.inf], # 범위 나누기
                         labels=['baby', '10대', '20대', '30대', '40대', '50대', '60대', 'old'], # 이름 붙이기
                         right=False)
                         
# 강사님
bin_cut = np.array([0, 9 , 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare["age"], 
                        bins=bin_cut, 
                        labels = (np.arange(12)*10).astype(str)+"대"))
welfare["age_group"]

ageg_income = welfare.dropna(subset = "income")\ 
                    .groupby("age_group", as_index = False)\
                    .agg(mean_income = ("income", "mean"))
sns.barplot(data = ageg_income, x="age_group", y="mean_income")
plt.show()
plt.clf()

# 판다스 데이터 프레임을 다룰때, 변수의 타입이 카테고리로 설정되어 있는 경우 groupby + agg 안 먹힘
welfare["age_group"] = welfare["age_group"].astype("object")
sex_age_income = welfare.dropna(subset = "income")\ 
                    .groupby(["age_group", "sex"], as_index = False)\
                    .agg(mean_income = ("income", "mean"))
sns.barplot(data=sex_age_income, x="age_group", y="mean_income", hue="sex")
plt.show()
plt.clf()

# 연령대별, 성별 상위 4% 수입 찾아보세요!
np.mean(np.arange(10)) # 벡터에 mean 씌운 것
x=np.arange(10)
# np.quantile(array, q=숫자(무조건 들어가야 함))
np.quantile(x, q=0.95) # quantile <- ppf 같은 존재 

# 중요!!!! 복습 많이
welfare["age_group"] = welfare["age_group"].astype("object")
sex_age_income_top4er = welfare.dropna(subset = "income")\ 
                    .groupby(["age_group", "sex"], as_index = False)\
                    .agg(top4per_income = ("income", lambda x: np.quantile(x, q = 0.96)))

# 다른 방법 
def my_f(vec):
    return vec.sum()
sex_age_income = welfare.dropna(subset = "income")\
                    .groupby(["age_group", "sex"], as_index = False)\
                    .agg(top4per_income = ("income", lambda x: my_f(x)))                    
