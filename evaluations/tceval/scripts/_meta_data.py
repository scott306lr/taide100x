
mmlu_tuples = [
    ['STEM', 'physics', 'astronomy'],
    ['STEM', 'physics', 'college_physics'],
    ['STEM', 'physics', 'conceptual_physics'],
    ['STEM', 'physics', 'high_school_physics'],
    ['STEM', 'chemistry', 'college_chemistry'],
    ['STEM', 'chemistry', 'high_school_chemistry'],
    ['STEM', 'biology', 'college_biology'],
    ['STEM', 'biology', 'high_school_biology'],
    ['STEM', 'computer science', 'college_computer_science'],
    ['STEM', 'computer science', 'computer_security'],
    ['STEM', 'computer science', 'high_school_computer_science'],
    ['STEM', 'computer science', 'machine_learning'],
    ['STEM', 'math', 'abstract_algebra'],
    ['STEM', 'math', 'college_mathematics'],
    ['STEM', 'math', 'elementary_mathematics'],
    ['STEM', 'math', 'high_school_mathematics'],
    ['STEM', 'math', 'high_school_statistics'],
    ['STEM', 'engineering', 'electrical_engineering'],
    ['humanities', 'history', 'high_school_european_history'],
    ['humanities', 'history', 'high_school_us_history'],
    ['humanities', 'history', 'high_school_world_history'],
    ['humanities', 'history', 'prehistory'],
    ['humanities', 'philosophy', 'formal_logic'],
    ['humanities', 'philosophy', 'logical_fallacies'],
    ['humanities', 'philosophy', 'moral_disputes'],
    ['humanities', 'philosophy', 'moral_scenarios'],
    ['humanities', 'philosophy', 'philosophy'],
    ['humanities', 'philosophy', 'world_religions'],
    ['humanities', 'law', 'international_law'],
    ['humanities', 'law', 'jurisprudence'],
    ['humanities', 'law', 'professional_law'],
    ['social sciences', 'politics', 'high_school_government_and_politics'],
    ['social sciences', 'politics', 'public_relations'],
    ['social sciences', 'politics', 'security_studies'],
    ['social sciences', 'politics', 'us_foreign_policy'],
    ['social sciences', 'culture', 'human_sexuality'],
    ['social sciences', 'culture', 'sociology'],
    ['social sciences', 'economics', 'econometrics'],
    ['social sciences', 'economics', 'high_school_macroeconomics'],
    ['social sciences', 'economics', 'high_school_microeconomics'],
    ['social sciences', 'geography', 'high_school_geography'],
    ['social sciences', 'psychology', 'high_school_psychology'],
    ['social sciences', 'psychology', 'professional_psychology'],
    ['other (business, health, misc.)', 'other', 'global_facts'],
    ['other (business, health, misc.)', 'other', 'miscellaneous'],
    ['other (business, health, misc.)', 'other', 'professional_accounting'],
    ['other (business, health, misc.)', 'business', 'business_ethics'],
    ['other (business, health, misc.)', 'business', 'management'],
    ['other (business, health, misc.)', 'business', 'marketing'],
    ['other (business, health, misc.)', 'health', 'anatomy'],
    ['other (business, health, misc.)', 'health', 'clinical_knowledge'],
    ['other (business, health, misc.)', 'health', 'college_medicine'],
    ['other (business, health, misc.)', 'health', 'human_aging'],
    ['other (business, health, misc.)', 'health', 'medical_genetics'],
    ['other (business, health, misc.)', 'health', 'nutrition'],
    ['other (business, health, misc.)', 'health', 'professional_medicine'],
    ['other (business, health, misc.)', 'health', 'virology']
]

mmlu_subject2category = {
    subject: category for category, subcategory, subject in mmlu_tuples
}

tmmluplus_tuples = [
    ['STEM', 'physics', 'physics', '物理'],
    ['STEM', 'physics', 'secondary_physics', '高中物理'],
    ['STEM', 'chemistry', 'organic_chemistry', '有機化學'],
    ['STEM', 'chemistry', 'advance_chemistry', '化學'],
    ['STEM', 'chemistry', 'junior_chemistry', '國中理化'],
    ['STEM', 'biology', 'pharmacy', '藥劑學'],
    ['STEM', 'biology', 'basic_medical_science', '基礎醫學'],
    ['STEM', 'biology', 'junior_science_exam', '國中會考基測自然科'],
    ['STEM', 'biology', 'tve_natural_sciences', '統測自然科'],
    ['STEM', 'computer science', 'computer_science', '資訊工程'],
    ['STEM', 'math', 'junior_math_exam', '國中會考基測數學科'],
    ['STEM', 'math', 'tve_mathematics', '統測數學'],
    ['STEM', 'math', 'engineering_math', '工程數學'],
    ['STEM', 'engineering', 'statistics_and_machine_learning', '統計與機器學習'],
    ['humanities', 'philosophy', 'jce_humanities', '指考人文科目'],
    ['humanities', 'law', 'general_principles_of_law', '法學大意'],
    ['humanities', 'law', 'anti_money_laundering', '洗錢防制'],
    ['humanities', 'law', 'introduction_to_law', '法律概論'],
    ['humanities', 'law', 'taxation', '稅務'],
    ['humanities', 'law', 'trust_practice', '信託實務'],
    ['humanities', 'law', 'administrative_law', '行政法'],
    ['social sciences', 'politics', 'national_protection', '軍事'],
    ['social sciences', 'politics', 'politic_science', '政治'],
    ['social sciences', 'culture', 'ttqav2', '台灣在地用語'],
    ['social sciences', 'culture', 'chinese_language_and_literature', '國文'],
    ['social sciences', 'culture', 'junior_chinese_exam', '國中會考基測國文'],
    ['social sciences', 'culture', 'tve_chinese_language', '統測國文'],
    ['social sciences', 'culture', 'three_principles_of_people', '三民主義'],
    ['social sciences', 'culture', 'taiwanese_hokkien', '閩南語'],
    ['social sciences', 'economics', 'economics', '經濟學'],
    ['social sciences', 'economics', 'macroeconomics', '總經'],
    ['social sciences', 'geography', 'geography_of_taiwan', '台灣地理'],
    ['social sciences', 'psychology', 'clinical_psychology', '臨床心理學'],
    ['social sciences', 'psychology', 'human_behavior', '人類行為與社會'],
    ['social sciences', 'psychology', 'educational_psychology', '教育心理'],
    ['social sciences', 'psychology', 'occupational_therapy_for_psychological_disorders', '心理障礙職能治療學'],
    ['other (business, health, misc.)', 'other', 'technical', '技術工相關'],
    ['other (business, health, misc.)', 'other', 'culinary_skills', '餐旅'],
    ['other (business, health, misc.)', 'other', 'mechanical', '機械與機電概論'],
    ['other (business, health, misc.)', 'other', 'logic_reasoning', '邏輯思維'],
    ['other (business, health, misc.)', 'other', 'real_estate', '房地產'],
    ['other (business, health, misc.)', 'other', 'marketing_management', '行銷管理'],
    ['other (business, health, misc.)', 'other', 'business_management', '企業管理'],
    ['other (business, health, misc.)', 'other', 'agriculture', '農業'],
    ['other (business, health, misc.)', 'other', 'official_document_management', '機關文書'],
    ['other (business, health, misc.)', 'other', 'fire_science', '火災學'],
    ['other (business, health, misc.)', 'other', 'optometry', '視光學'],
    ['other (business, health, misc.)', 'other', 'insurance_studies', '保險學'],
    ['other (business, health, misc.)', 'other', 'nautical_science', '航海'],
    ['other (business, health, misc.)', 'other', 'tve_design', '統測＿設計'],
    ['other (business, health, misc.)', 'other', 'junior_social_studies', '國中會考基測社會科'],
    ['other (business, health, misc.)', 'other', 'music', '音樂科'],
    ['other (business, health, misc.)', 'business', 'finance_banking', '金融與法規'],
    ['other (business, health, misc.)', 'business', 'financial_analysis', '財務分析'],
    ['other (business, health, misc.)', 'business', 'management_accounting', '管理會計'],
    ['other (business, health, misc.)', 'business', 'accounting', '會計學'],
    ['other (business, health, misc.)', 'business', 'auditing', '審計學'],
    ['other (business, health, misc.)', 'business', 'trade', '貿易'],
    ['other (business, health, misc.)', 'health', 'dentistry', '牙醫學'],
    ['other (business, health, misc.)', 'health', 'traditional_chinese_medicine_clinical_medicine', '中醫臨床醫學'],
    ['other (business, health, misc.)', 'health', 'veterinary_pathology', '獸醫病理學'],
    ['other (business, health, misc.)', 'health', 'pharmacology', '藥理學'],
    ['other (business, health, misc.)', 'health', 'veterinary_pharmacology', '獸醫藥理學'],
    ['social sciences', 'education', 'education', '教育'],
    ['social sciences', 'education', 'education_(profession_level)', '教育_技術'],
    ['social sciences', 'education', 'physical_education','體育']
]

tmmluplus_subject2category = {
    subject: category for category, subcategory, subject, name in tmmluplus_tuples
}