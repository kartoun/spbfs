# spbfs.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

def propensity_score_matching(df, outcome_var, match_features, caliper=0.1, m=1):
    X = df[match_features]
    y = df[outcome_var]

    model = LogisticRegression()
    model.fit(X, y)
    pscore = model.predict_proba(X)[:, 1]
    df = df.copy()
    df['pscore'] = pscore

    treated = df[df[outcome_var] == 1]
    control = df[df[outcome_var] == 0]

    matched_control_indices = []

    for idx, row in treated.iterrows():
        distance = np.abs(control['pscore'] - row['pscore'])
        eligible = distance[distance <= caliper]
        if len(eligible) >= m:
            matched_indices = eligible.nsmallest(m).index
            matched_control_indices.extend(matched_indices)

    matched_df = pd.concat([treated, control.loc[matched_control_indices]]).drop(columns=['pscore'])
    return matched_df

def get_selected_features(data, feature_names, outcome_var,
                          num_iterations=100,
                          num_random_variables_for_matching=3,
                          final_selection_threshold=0.5,
                          caliper_value=0.1,
                          m_value=1,
                          p_value_threshold=0.001,
                          verbose=0):

    selected_features_counts = {feature: 0 for feature in feature_names}

    for iteration in range(num_iterations):
        sampled_features = np.random.choice(feature_names,
                                            num_random_variables_for_matching,
                                            replace=False)

        matched_df = propensity_score_matching(data,
                                               outcome_var,
                                               sampled_features,
                                               caliper=caliper_value,
                                               m=m_value)

        unmatched_features = [f for f in feature_names if f not in sampled_features]

        for feature in unmatched_features:
            group1 = matched_df[matched_df[outcome_var] == 1][feature]
            group0 = matched_df[matched_df[outcome_var] == 0][feature]

            _, p_value = ttest_ind(group1, group0, equal_var=False)

            if p_value < p_value_threshold:
                selected_features_counts[feature] += 1

        if verbose:
            print(f"Completed iteration {iteration + 1}/{num_iterations}")

    # Calculate frequency of selection
    results = [(feature, count / num_iterations) for feature, count in selected_features_counts.items()]
    results_df = pd.DataFrame(results, columns=['Feature_Name', 'Frequency'])

    # Apply final threshold
    results_df = results_df[results_df['Frequency'] > final_selection_threshold]
    results_df = results_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

    return results_df