import os
import pandas as pd


class MetaDataManager:
    def __init__(self, metadata_csv_path: str):
        """
        메타데이터 CSV 파일 경로를 받아 데이터프레임으로 로드한다.
        """
        self.metadata_df = pd.read_csv(metadata_csv_path)

    def get_metadata(self, image_name: str) -> tuple:
        """
        주어진 이미지 이름에 해당하는 메타데이터를 반환한다.
        """
        image_id_str = os.path.dirname(image_name)  # 'ID091'
        image_id_num = image_id_str.lstrip('ID').lstrip('0')  
        try:
            image_id = int(image_id_num)  # 정수로 변환
        except ValueError:
            return 'Unknown', 'Unknown', 'Unknown', 'Unknown', 'unknown'

        metadata_row = self.metadata_df[self.metadata_df['ID'] == image_id]
        if not metadata_row.empty:
            age = str(metadata_row['Age'].values[0])
            gender = metadata_row['Gender'].values[0]
            weight = str(metadata_row['Weight (kg)'].values[0])
            height = str(metadata_row['Height (cm)'].values[0])
        else:
            age = 'Unknown'
            gender = 'Unknown'
            weight = 'Unknown'
            height = 'Unknown'

        bone_descriptor = self.get_bone_descriptor(age, gender)

        return age, gender, weight, height, bone_descriptor

    @staticmethod
    def get_bone_descriptor(age: str, gender: str) -> str:

        try:
            age = int(age)
        except ValueError:
            return 'unknown'

        gender = gender.lower()

        if age < 13:
            return 'developing'  # 어린이
        elif age < 20:
            return 'rapidly growing'  # 청소년
        elif age < 60:
            if gender == 'male':
                return 'dense and thick'  # 성인 남성
            elif gender == 'female':
                return 'slender and delicate'  # 성인 여성
            else:
                return 'adult'
        else:
            return 'aging'  # 노인
