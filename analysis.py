import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

datapath_1 = r'dataset.csv'
datapath_2 = r'Symptom-severity.csv'

try:
    df1 = pd.read_csv(datapath_1)
    df2 = pd.read_csv(datapath_2)
    print("Đọc file thành công")

    symptom_col = [col for col in df1.columns if col != 'Disease']

    
    df1.iloc[:, 1:] = df1.iloc[:, 1:].map(lambda x: x.strip() if isinstance(x, str) else x)
    df2['Symptom'] = df2['Symptom'].str.strip()

    typo_map = {
        'foul_smell_of urine': 'foul_smell_ofurine',
        'dischromic _patches': 'dischromic_patches',
        'spotting_ urination': 'spotting_urination'
    }
    df1.iloc[:, 1:] = df1.iloc[:, 1:].replace(typo_map)
    df1_clean = df1.copy()
    df1.fillna('None', inplace=True)
    df2.loc[len(df2)] = ['None', 0]

    print("Tiền xử lý hoàn tất")
    print("Số bệnh unique:", df1['Disease'].nunique())
    print("Số triệu chứng unique (df2):", df2['Symptom'].nunique())

    # Top 10 bệnh phổ biến nhất
    top_diseases = df1['Disease'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top_diseases.index, top_diseases.values)
    plt.xlabel('Số lượng')
    plt.title('Top 10 bệnh phổ biến nhất')
    plt.tight_layout()
    plt.show()

    # Top 10 triệu chứng phổ biến nhất
    all_symptoms_list = []
    for col in symptom_col:
        for val in df1[col]:
            if val != 'None':
                all_symptoms_list.append(val)
    symptom_counts = pd.Series(all_symptoms_list).value_counts().head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(symptom_counts.index, symptom_counts.values)
    plt.xlabel('Số lần xuất hiện')
    plt.title('Top 10 triệu chứng phổ biến nhất')
    plt.tight_layout()
    plt.show()

    # Phân phối số triệu chứng mỗi bệnh nhân
    symptom_count_per_patient = []
    for _, row in df1.iterrows():
        count = sum(1 for col in symptom_col if row[col] != 'None')
        symptom_count_per_patient.append(count)
    plt.figure(figsize=(8, 5))
    plt.hist(symptom_count_per_patient, bins=17, edgecolor='black')
    plt.xlabel('Số triệu chứng')
    plt.ylabel('Số bệnh nhân')
    plt.title('Phân phối số triệu chứng mỗi bệnh nhân')
    plt.tight_layout()
    plt.show()

    disease_list = df1['Disease'].unique().tolist()
    transaction = []
    for _, row in df1.iterrows():
        items = [row[col] for col in symptom_col if row[col] != 'None']
        items.append(row['Disease'])
        transaction.append(items)

    te = TransactionEncoder()
    te_array = te.fit_transform(transaction)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    print("\ndf_encoded shape:", df_encoded.shape)

    frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True, max_len=4)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
    rules = rules.sort_values('lift', ascending=False)
    print("\nTop 20 rules theo lift:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20).to_string())

    # Bar chart top 20 rules theo lift
    top_rules = rules.head(20).copy()
    top_rules['rule'] = (
        top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        + ' → '
        + top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
    )
    plt.figure(figsize=(12, 8))
    plt.barh(top_rules['rule'], top_rules['lift'])
    plt.xlabel('Lift')
    plt.title('Top 20 Association Rules theo Lift')
    plt.tight_layout()
    plt.show()

    # Scatter support vs confidence
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(rules['support'], rules['confidence'], alpha=0.5, c=rules['lift'], cmap='viridis')
    plt.colorbar(sc, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (màu = Lift)')
    plt.tight_layout()
    plt.show()

    # Lọc rules triệu chứng → bệnh
    rules_disease = rules[rules['consequents'].apply(
        lambda x: len(x) == 1 and list(x)[0] in disease_list
    )].copy()
    rules_disease = rules_disease.sort_values('confidence', ascending=False)
    print("\nSố rules triệu chứng → bệnh:", len(rules_disease))
    print(rules_disease[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20).to_string())

    # Bar chart top 20 rules triệu chứng → bệnh
    top_rules_disease = rules_disease.head(20).copy()
    top_rules_disease['rule'] = (
        top_rules_disease['antecedents'].apply(lambda x: ', '.join(list(x)))
        + ' → '
        + top_rules_disease['consequents'].apply(lambda x: ', '.join(list(x)))
    )
    plt.figure(figsize=(12, 8))
    plt.barh(top_rules_disease['rule'], top_rules_disease['confidence'])
    plt.xlabel('Confidence')
    plt.title('Top 20 Rules: Triệu chứng → Bệnh')
    plt.tight_layout()
    plt.show()

    severity_dict = dict(zip(df2['Symptom'], df2['weight']))

    df_weighted = df1.copy()
    for col in symptom_col:
        df_weighted[col] = df_weighted[col].map(lambda x: severity_dict.get(x, 0))

    le = LabelEncoder()
    df_weighted['Disease'] = le.fit_transform(df_weighted['Disease'])

    X = df_weighted[symptom_col].astype(float)
    y = df_weighted['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    report = classification_report(y_test, y_pred, target_names=le.classes_,
                                   zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_diseases = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])
    df_diseases = df_diseases[['precision', 'recall', 'f1-score', 'support']].round(2)

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis('off')

    table = ax.table(
        cellText=df_diseases.values,
        rowLabels=df_diseases.index,
        colLabels=['Precision', 'Recall', 'F1-score', 'Support'],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    feature_names_real = []
    for col in symptom_col:
        valid_vals = df1_clean[col].dropna()
        feature_names_real.append(
            valid_vals.value_counts().index[0] if len(valid_vals) > 0 else col
        )

    plt.figure(figsize=(20, 10))
    plot_tree(clf, max_depth=3, feature_names=feature_names_real,
              class_names=le.classes_, filled=True, fontsize=6,
              impurity=False, rounded=True)
    plt.title('Decision Tree - Dự đoán bệnh từ triệu chứng (3 tầng đầu)')
    plt.tight_layout()
    plt.show()

    print("\nHoàn tất!")

except FileNotFoundError:
    print("Không tìm thấy file")