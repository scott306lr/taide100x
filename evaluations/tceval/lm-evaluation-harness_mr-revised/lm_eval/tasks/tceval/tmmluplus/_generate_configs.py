"""
Take in a YAML, and output all other splits with this YAML
"""
import os
import yaml
import argparse

from tqdm import tqdm

SUBJECTS = ['tmmluplus-accounting', 'tmmluplus-administrative_law', 'tmmluplus-advance_chemistry', 'tmmluplus-agriculture', 'tmmluplus-anti_money_laundering', 'tmmluplus-auditing', 'tmmluplus-basic_medical_science', 'tmmluplus-business_management', 'tmmluplus-chinese_language_and_literature', 'tmmluplus-clinical_psychology', 'tmmluplus-computer_science', 'tmmluplus-culinary_skills', 'tmmluplus-dentistry', 'tmmluplus-economics', 'tmmluplus-education', 'tmmluplus-education_(profession_level)', 'tmmluplus-educational_psychology', 'tmmluplus-engineering_math', 'tmmluplus-finance_banking', 'tmmluplus-financial_analysis', 'tmmluplus-fire_science', 'tmmluplus-general_principles_of_law', 'tmmluplus-geography_of_taiwan', 'tmmluplus-human_behavior', 'tmmluplus-insurance_studies', 'tmmluplus-introduction_to_law', 'tmmluplus-jce_humanities', 'tmmluplus-junior_chemistry', 'tmmluplus-junior_chinese_exam', 'tmmluplus-junior_math_exam', 'tmmluplus-junior_science_exam', 'tmmluplus-junior_social_studies', 'tmmluplus-logic_reasoning', 'tmmluplus-macroeconomics', 'tmmluplus-management_accounting', 'tmmluplus-marketing_management', 'tmmluplus-mechanical', 'tmmluplus-music', 'tmmluplus-national_protection', 'tmmluplus-nautical_science', 'tmmluplus-occupational_therapy_for_psychological_disorders', 'tmmluplus-official_document_management', 'tmmluplus-optometry', 'tmmluplus-organic_chemistry', 'tmmluplus-pharmacology', 'tmmluplus-pharmacy', 'tmmluplus-physical_education', 'tmmluplus-physics', 'tmmluplus-politic_science', 'tmmluplus-real_estate', 'tmmluplus-secondary_physics', 'tmmluplus-statistics_and_machine_learning', 'tmmluplus-taiwanese_hokkien', 'tmmluplus-taxation', 'tmmluplus-technical', 'tmmluplus-three_principles_of_people', 'tmmluplus-trade', 'tmmluplus-traditional_chinese_medicine_clinical_medicine', 'tmmluplus-trust_practice', 'tmmluplus-ttqav2', 'tmmluplus-tve_chinese_language', 'tmmluplus-tve_design', 'tmmluplus-tve_mathematics', 'tmmluplus-tve_natural_sciences', 'tmmluplus-veterinary_pathology', 'tmmluplus-veterinary_pharmacology']

def get_second_group(name):
    cat2subjects = {
        "STEM":[
            "organic_chemistry",
            "advance_chemistry",
            "physics",
            "secondary_physics",
            "linear_algebra",
            "pharmacy",
            "statistics_and_machine_learning",
            "computer_science",
            "basic_medical_science",
            "junior_science_exam",
            "junior_math_exam",
            "tve_mathematics",
            "tve_natural_sciences",
            "junior_chemistry",
            "engineering_math"
        ],
        "other":[
            "dentistry",
            "traditional_chinese_medicine_clinical_medicine",
            "technical",
            "culinary_skills",
            "mechanical",
            "logic_reasoning",
            "real_estate",
            "finance_banking",
            "marketing_management",
            "business_management",
            "agriculture",
            "official_document_management",
            "financial_analysis",
            "management_accounting",
            "veterinary_pathology",
            "accounting",
            "fire_science",
            "optometry",
            "insurance_studies",
            "pharmacology",
            "veterinary_pharmacology",
            "nautical_science",
            "auditing",
            "trade",
            "tve_design",
            "junior_social_studies",
            "music"
        ],
        "humanities":[
            "general_principles_of_law",
            "anti_money_laundering",
            "jce_humanities",
            "introduction_to_law",
            "taxation",
            "trust_practice",
            "administrative_law"
        ],
        "social sciences":[
            "clinical_psychology",
            "ttqav2",
            "human_behavior",
            "national_protection",
            "politic_science",
            "educational_psychology",
            "education_(profession_level)",
            "economics",
            "occupational_therapy_for_psychological_disorders",
            "geography_of_taiwan",
            "physical_education",
            "macroeconomics",
            "chinese_language_and_literature",
            "junior_chinese_exam",
            "tve_chinese_language",
            "education",
            "three_principles_of_people",
            "taiwanese_hokkien"
        ]
    }
        
    for c, t in cat2subjects.items():
        if name.strip('tmmluplus-') in t:
            return f'{c.replace(" ", "_")}'
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot',type=str,default='fewshot')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = "_default_template_yaml"
    with open(os.path.join(args.shot, base_yaml_name)) as f:
        base_yaml = yaml.full_load(f)


    for subject_eng in SUBJECTS:
        task_name = subject_eng.replace('-',f'_{args.shot}-')

        if not get_second_group(subject_eng):
            yaml_dict = {
                "group": ["tc-eval-v2",f"tmmluplus_{args.shot}"],
                "include": base_yaml_name,
                "task": task_name,
                "dataset_name": subject_eng,
            }
        else:
            yaml_dict = {
                "group": ["tc-eval-v2", f"tc-eval-v2_{args.shot}", f"tmmluplus_{args.shot}", f"tmmluplus_{args.shot}-"+get_second_group(subject_eng)],
                "include": base_yaml_name,
                "task": task_name,
                "dataset_name": subject_eng,
            }

        file_save_path = f"{subject_eng}.yaml"
        print(f"Saving yaml for subset {subject_eng} to {file_save_path}")
        with open(os.path.join(args.shot, file_save_path), "w") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )