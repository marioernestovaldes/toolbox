import missingno as msno
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import plotly.express as px
import shap

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GroupKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    # plot_confusion_matrix,
    mean_squared_error,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy

from matplotlib import pyplot as pl


def remove_outliers(df: pd.DataFrame):
    """
    Remove outliers from a dataframe
    """
    # The first round of outlier detection is using expression values
    from collections import Counter
    from sklearn.ensemble import IsolationForest
    from scipy.stats import zscore

    df['count'] = df.count(axis=1)
    # Calculate the z-score for the quantification count feature
    df['count'] = zscore(df['count'])

    iso_forest = IsolationForest(n_estimators=300, random_state=42)

    # outliers_count = iso_forest.fit_predict(df.applymap(lambda x: 1 if not pd.isnull(x) else 0))
    #
    # dict_outliers_count = dict(zip(df.index, outliers_count))
    #
    # print('Outlier removal based on protein count...')
    # print(f'Kepping {Counter(outliers_count)[1]} samples...')
    # print(f'Removing {Counter(outliers_count)[-1]} samples...')
    # print('')
    #
    # df = df.loc[[i for i in df.index if dict_outliers_count[i] == 1], :]

    outliers_expression = iso_forest.fit_predict(df.replace(np.nan, 0))

    dict_outliers_expression = dict(zip(df.index, outliers_expression))

    print('Outlier removal based on protein expression...')
    print(f'Kepping {Counter(outliers_expression)[1]} samples...')
    print(f'Removing {Counter(outliers_expression)[-1]} samples...')

    df = df.loc[[i for i in df.index if dict_outliers_expression[i] == 1], :].drop('count', axis=1)

    return df

def hierarchical_clustering(
        df,
        vmin=None,
        vmax=None,
        show="scaled",
        figsize=(8, 8),
        top_height=2,
        left_width=2,
        xmaxticks=None,
        ymaxticks=None,
        metric="euclidean",
        cmap=None,
        scaling="standard",
        scaling_kws=None,
):
    """Generates a heatmap with hierarchical clustering from the input matrix.

    Args:
        df (pd.DataFrame): The input matrix for which you want to create a heatmap.
        vmin (float): Optional parameter for setting the minimum value of the heatmap color scale.
        vmax (float): Optional parameter for setting the maximum value of the heatmap color scale.
        show (str): Specifies how the data should be displayed. Can be "scaled" or "original".
        figsize (tuple): A tuple specifying the size of the resulting heatmap figure.
        top_height (float): Height of the top component in the figure.
        left_width (float): Width of the left component in the figure.
        xmaxticks (int): Number of ticks on the x-axis.
        ymaxticks (int): Number of ticks on the y-axis.
        metric (str): The metric used for hierarchical clustering, with "euclidean" as the default.
        cmap (str): Colormap to use for the heatmap.
        scaling (str): Type of scaling to apply to the input data.
        scaling_kws (dict): Additional parameters for data scaling.

    Returns:
        pd.DataFrame: The clustered data frame.
        matplotlib.figure.Figure: The heatmap figure.
    """
    df_orig = df.copy()
    df = df.copy()

    if scaling is not None:
        if scaling_kws is None:
            scaling_kws = {}
        df = scale_dataframe(df, how=scaling, **scaling_kws)

    total_width, total_height = figsize
    main_h = 1 - (top_height / total_height)
    main_w = 1 - (left_width / total_width)

    gap_x = 0.1 / total_width
    gap_y = 0.1 / total_height

    left_h = main_h
    left_w = 1 - main_w

    top_h = 1 - main_h
    top_w = main_w

    ydim, xdim = df.shape

    if xmaxticks is None:
        xmaxticks = int(5 * main_w * total_width)
    if ymaxticks is None:
        ymaxticks = int(5 * main_h * total_height)

    dm = df.fillna(0).values

    D1 = squareform(pdist(dm, metric=metric))
    D2 = squareform(pdist(dm.T, metric=metric))

    fig = pl.figure(figsize=figsize)
    fig.set_tight_layout(False)

    # add left dendrogram
    ax1 = fig.add_axes([0, 0, left_w - gap_x, left_h], frameon=False)
    Y = linkage(D1, method="complete")
    Z1 = dendrogram(Y, orientation="left", color_threshold=0, above_threshold_color="k")
    ax1.set_xticks([])
    ax1.set_yticks([])
    # add top dendrogram
    ax2 = fig.add_axes([left_w, main_h + gap_y, top_w, top_h - gap_y], frameon=False)
    Y = linkage(D2, method="complete")
    Z2 = dendrogram(Y, color_threshold=0, above_threshold_color="k")
    ax2.set_xticks([])
    ax2.set_yticks([])
    # add matrix plot
    axmatrix = fig.add_axes([left_w, 0, main_w, main_h])
    idx1 = Z1["leaves"]
    idx2 = Z2["leaves"]

    if show == "scaled":
        D = dm[idx1, :]
        D = D[:, idx2]
    if show == "original":
        D = df_orig.iloc[idx1, :]
        D = D.iloc[:, idx2].values

    if cmap is None:
        cmap = "hot"
    fig = axmatrix.matshow(D[::-1], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    ax = pl.gca()
    ax.yaxis.tick_right()
    ax.xaxis.tick_bottom()

    clustered = df_orig.iloc[Z1["leaves"][::-1], Z2["leaves"]]

    ndx_y = np.linspace(0, len(clustered.index) - 1, ymaxticks)
    ndx_x = np.linspace(0, len(clustered.columns) - 1, xmaxticks)
    ndx_y = [int(i) for i in ndx_y]
    ndx_x = [int(i) for i in ndx_x]

    _ = pl.yticks(ndx_y, clustered.iloc[ndx_y].index)
    _ = pl.xticks(ndx_x, clustered.columns[ndx_x], rotation=90)

    return clustered, fig


def scale_dataframe(df, how="standard", **kwargs):
    """
    Scales a DataFrame using the specified method.

    Args:
        df (pd.DataFrame): The DataFrame to be scaled.
        how (str): The scaling method to use, can be "standard" or "robust".
        **kwargs: Additional keyword arguments for the scaler.

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    if how == "standard":
        scaler = StandardScaler
    elif how == "robust":
        scaler = RobustScaler
    df = df.copy()
    df.loc[:, :] = scaler(**kwargs).fit_transform(df)
    return df


def plot_missing_values(df, kind="matrix"):
    """
    Plots missing values in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        kind (str): The type of plot to create, can be "matrix", "bar", or "heatmap".

    Returns:
        msno.matrix, msno.bar, or None: The missing value plot or None if "heatmap" is selected.
    """
    if kind == "matrix":
        return msno.matrix(df)
    if kind == "bar":
        return msno.bar(df)
    if kind == "heatmap":
        msno.heatmap(df)


def knn_score(df, var_names, tgt_name, **params):
    """
    Calculate the k-Nearest Neighbors (KNN) accuracy of a clustered dataset with known labels.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.
        var_names (list): The list of feature variable names to use for KNN.
        tgt_name (str): The name of the target variable (known labels).
        **params: Additional keyword arguments for the KNeighborsClassifier.

    Returns:
        float: The accuracy score of the KNN model.
        pd.DataFrame: The confusion matrix.
        pd.DataFrame: The predictions made by the KNN model.
    """
    df = df.copy().reset_index(drop=True)

    X = df[var_names]
    y = df[tgt_name]

    kfold = KFold(n_splits=len(X))

    prediction = y.copy().to_frame()
    for ndx_train, ndx_valid in kfold.split(X):
        X_train, X_valid = X.loc[ndx_train], X.loc[ndx_valid]
        y_train, y_valid = y.loc[ndx_train], y.loc[ndx_valid]

        clf = KNeighborsClassifier(**params)
        clf.fit(X_train, y_train)
        prd = clf.predict(X_valid)

        prediction.loc[ndx_valid, "Prediction"] = prd

    classes = y.value_counts().index
    conf_ma = confusion_matrix(
        prediction[tgt_name], prediction["Prediction"], labels=classes
    )
    df_coma = pd.DataFrame(conf_ma, index=classes, columns=classes)
    accuracy = accuracy_score(prediction[tgt_name], prediction["Prediction"])

    return accuracy, df_coma, prediction


def sklearn_cv_classification(
        X, y, base_model, params={}, X_test=None, n_folds=5, seeds=None
):
    """
    Perform cross-validation for a classification model using scikit-learn.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target labels.
        base_model (class): The scikit-learn model class to use.
        params (dict): Additional keyword arguments for the model.
        X_test (pd.DataFrame): Test data for predictions (optional).
        n_folds (int): The number of cross-validation folds.
        seeds (list): List of random seeds for reproducibility.

    Returns:
        float: Mean accuracy of cross-validation.
        float: Standard deviation of accuracy scores.
        pd.DataFrame: Predictions from cross-validation.
        pd.DataFrame: Predictions on the test data (if provided).
    """
    if seeds is None:
        seeds = [1]

    losses = []
    predic = None

    cv_predictions = X[[]].copy()
    cv_predictions["CV-pred"] = None
    cv_predictions["y"] = y

    n_models = len(seeds) * n_folds

    if X_test is not None:
        predictions = X_test[[]].copy()

    for seed in seeds:

        assert isinstance(seed, int)

        params["seed"] = seed

        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)

        for _n, (ndx_train, ndx_valid) in enumerate(kfold.split(X, y)):

            print("=" * 22 + f" Fold {_n:3d} " + "=" * 22)

            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y[ndx_train], y[ndx_valid]

            _model = base_model()
            _model.fit(_X_train, _y_train)

            _pred = _model.predict(_X_valid)
            _loss = accuracy_score(_y_valid, _pred)

            cv_predictions.iloc[ndx_valid, 0] = _pred

            print(f"Fold {_n} accuracy: {_loss}")

            losses.append(_loss)

            if X_test is not None:
                _pred_test = _model.predict(X_test)
                predictions[f"seed-{seed}-fold-{_n}"] = _pred_test.astype(int)

    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_predictions = cv_predictions.astype(int)
    print(f"CV-loss {loss_mean}+/-{loss_std}")

    return loss_mean, loss_std, cv_predictions, predictions


def sklearn_cv_binary_clf_roc(
        X,
        y,
        base_model,
        params={},
        X_test=None,
        n_folds=5,
        seeds=None,
        to_numpy=False,
        metric=None,
        fit_kws=None,
        framework=None,
):
    """
    Perform cross-validation for a binary classification model using scikit-learn.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The binary target labels (0 or 1).
        base_model (class): The scikit-learn binary classification model class.
        params (dict): Additional keyword arguments for the model.
        X_test (pd.DataFrame): Test data for predictions (optional).
        n_folds (int): The number of cross-validation folds.
        seeds (list): List of random seeds for reproducibility.
        to_numpy (bool): Convert data to NumPy arrays if True.
        metric (callable): The scoring metric function, e.g., roc_auc_score.
        fit_kws (dict): Additional keyword arguments for the model's fit method.
        framework (str): Specify the framework for specific model adjustments, e.g., "lgbm".

    Returns:
        float: Mean ROC AUC of cross-validation.
        float: Standard deviation of ROC AUC scores.
        pd.DataFrame: Predictions from cross-validation.
        pd.DataFrame: Predictions on the test data (if provided).
    """
    if seeds is None:
        seeds = [1]
    losses = []
    predic = None
    cv_predictions = X[[]].copy()
    cv_predictions["y"] = y
    if metric is None:
        metric = roc_auc_score
    if fit_kws is None:
        fit_kws = {}
    n_models = len(seeds) * n_folds
    if X_test is not None:
        predictions = X_test[[]].copy()
    else:
        predictions = None
    for n_seed, seed in enumerate(seeds):
        print("+" * 22 + f" Seed {n_seed:2d} " + "+" * 23)
        assert isinstance(seed, int)
        params["seed"] = seed
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for n_fold, (ndx_train, ndx_valid) in enumerate(kfold.split(X, y)):
            print("=" * 22 + f" Fold {n_fold:2d} " + "=" * 23)
            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y.iloc[ndx_train], y.iloc[ndx_valid]
            if to_numpy:
                _X_train, _X_valid, _y_train, _y_valid = (
                    _X_train.values,
                    _X_valid.values,
                    _y_train.values,
                    _y_valid.values,
                )
            _model = base_model(**params)
            if framework == "lgbm":
                fit_kws["eval_set"] = [(_X_train, _y_train), (_X_valid, _y_valid)]
                fit_kws["eval_metric"] = "auc"
                fit_kws["eval_names"] = ["Train", "Valid"]
            _model.fit(_X_train, _y_train, **fit_kws)
            _pred = _model.predict_proba(_X_valid)[:, 1]
            _loss = metric(_y_valid, _pred)
            print(f"Fold {n_fold}: {_loss:1.4f}")
            cv_predictions.loc[ndx_valid, f"cv-pred-{n_seed}"] = _pred
            losses.append(_loss)
            if X_test is not None:
                _pred_test = _model.predict_proba(X_test.values)[:, 1]
                predictions[f"pred-s-{seed}"] = _pred_test
    cv_predictions["cv-pred"] = cv_predictions.filter(regex="cv-pred").mean(axis=1)
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_loss = metric(cv_predictions["y"].values, cv_predictions["cv-pred"].values)
    if X_test is not None:
        predictions["pred"] = predictions.filter(regex="pred").mean(axis=1)
    print(f"Avg-CV: {loss_mean:1.4f}+/-{loss_std:1.4f}; Final-CV: {cv_loss:1.4f}")
    return loss_mean, loss_std, cv_predictions, predictions


def tabnet_cv_classification(
        X,
        y,
        base_model,
        params={},
        X_test=None,
        n_folds=5,
        seeds=None,
        to_numpy=False,
        fit_kws=None,
        metric=None,
):
    """
    Perform cross-validation for a binary classification model using TabNet.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The binary target labels (0 or 1).
        base_model (class): The TabNet binary classification model class.
        params (dict): Additional keyword arguments for the model.
        X_test (pd.DataFrame): Test data for predictions (optional).
        n_folds (int): The number of cross-validation folds.
        seeds (list): List of random seeds for reproducibility.
        to_numpy (bool): Convert data to NumPy arrays if True.
        fit_kws (dict): Additional keyword arguments for the model's fit method.
        metric (callable): The scoring metric function, e.g., roc_auc_score.

    Returns:
        float: Mean ROC AUC of cross-validation.
        float: Standard deviation of ROC AUC scores.
        pd.DataFrame: Predictions from cross-validation.
        pd.DataFrame: Predictions on the test data (if provided).
    """
    if seeds is None:
        seeds = [1]
    if fit_kws is None:
        fit_kws = {}
    if metric is None:
        metric = roc_auc_score
    losses = []
    predic = None
    cv_predictions = X[[]].copy()
    cv_predictions["y"] = y
    n_models = len(seeds) * n_folds
    if X_test is not None:
        predictions = X_test[[]].copy()
    else:
        predictions = None
    for n_seed, seed in enumerate(seeds):
        assert isinstance(seed, int)
        print("+" * 22 + f" Seed {n_seed:2d} " + "+" * 23)
        params["seed"] = seed
        kfold = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
        for n_fold, (ndx_train, ndx_valid) in enumerate(kfold.split(X, y)):
            print("=" * 22 + f" Fold {n_fold:2d} " + "=" * 23)
            _X_train, _X_valid = X.iloc[ndx_train], X.iloc[ndx_valid]
            _y_train, _y_valid = y.iloc[ndx_train], y.iloc[ndx_valid]
            if to_numpy:
                _X_train, _X_valid, _y_train, _y_valid = (
                    _X_train.values,
                    _X_valid.values,
                    _y_train.values,
                    _y_valid.values,
                )
            _model = base_model(seed=seed)
            _model.fit(
                _X_train,
                _y_train,
                eval_set=[(_X_train, _y_train), (_X_valid, _y_valid)],
                eval_name=["train", "valid"],
                **fit_kws,
            )
            _pred = _model.predict_proba(_X_valid)[:, 1]
            _loss = metric(_y_valid, _pred)
            cv_predictions.loc[ndx_valid, f"cv-pred-{n_seed}"] = _pred
            print(f"Fold {n_fold}: {_loss}")
            losses.append(_loss)
            if X_test is not None:
                _pred_test = _model.predict_proba(X_test.values)[:, 1]
                predictions[f"pred-s-{seed}"] = _pred_test
            del _model
    loss_mean = np.mean(losses)
    loss_std = np.std(losses)
    cv_predictions["cv-pred"] = cv_predictions.filter(regex="cv-pred").mean(axis=1)
    cv_loss = metric(cv_predictions["y"].values, cv_predictions["cv-pred"].values)
    cv_predictions = cv_predictions.astype(float)
    if X_test is not None:
        predictions["pred"] = predictions.filter(regex="pred").mean(axis=1)
    print(f"Avg-CV: {loss_mean:1.4f}+/-{loss_std:1.4f}; Final-CV: {cv_loss:1.4f}")
    return loss_mean, loss_std, cv_predictions, predictions


def softmax(a, axis=None):
    """
    Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
    :param a: ndarray/tensor
    :param axis: axis to sum over; default (None) sums over everything

    Compute the softmax function for an array, which is often used in multiclass classification.

    Args:
        a: ndarray/tensor - The input array or tensor.
        axis: int - Axis to sum over; default (None) sums over everything.

    Returns:
        ndarray/tensor: The softmax values.
    """
    from scipy.special import logsumexp

    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    return np.exp(a - lse)


def decode_prediction(df, encoder):
    """
    Decode a one-hot encoded prediction using a given encoder.

    Args:
        df: pd.DataFrame - The one-hot encoded predictions.
        encoder: sklearn.preprocessing.LabelEncoder - The encoder to use for decoding.

    Returns:
        pd.DataFrame: The decoded predictions.
    """
    return df.apply(encoder.inverse_transform)


def remove_features_with_anti_target(
        df, features, anti_target="is_test", target_auc=0.6
):
    """
    Remove features based on anti-target performance and a target AUC threshold.

    Args:
        df: pd.DataFrame - The feature matrix.
        features: list - The list of features to consider.
        anti_target: str - The name of the anti-target variable.
        target_auc: float - The target AUC threshold.

    Returns:
        pd.DataFrame: History of feature selection.
        """
    features = copy(features)

    auc = 1

    history = dict(n_features=[], auc=[], features=[])

    while len(features) > 0:
        anti_target = "is_test"

        dtrain = xgb.DMatrix(df[features], df[anti_target])

        params = {
            "objective": "binary:logistic",
            "max_depth": 2,
        }

        cv = xgb.cv(dtrain=dtrain, params=params, metrics=["auc"])

        auc = cv["train-auc-mean"].mean()

        history["auc"].append(auc)
        history["n_features"].append(len(features))
        history["features"].append(copy(features))

        if auc < target_auc:
            break

        params["metric"] = "logloss"

        model = xgb.XGBClassifier(params=params, verbosity=0)
        model.fit(df[features], df[anti_target])

        fi = pd.DataFrame(
            {"Feature": features, "Importance": model.feature_importances_}
        )
        fi = fi.sort_values("Importance", ascending=False).reset_index(drop=True)

        best_feature = fi.loc[0, "Feature"]
        features.remove(best_feature)

    if len(features) == 1:
        logging.warning(
            "Target AUC could not be reached, returning last remaining feature."
        )

    return pd.DataFrame(history)


def quick_pca(df, n_components=2, labels=None, plot=True, scale=True, interactive=False, **plot_kws):
    """
    Quickly perform Principal Component Analysis (PCA) and optionally create a pairplot or scatter matrix.

    Args:
        df: pd.DataFrame - The input data for PCA.
        n_components: int - The number of components to keep.
        labels: pd.Series - Labels for coloring the plot (optional).
        plot: bool - If True, create a plot (pairplot or scatter matrix).
        scale: bool - If True, scale the data using StandardScaler.
        interactive: bool - If True, create an interactive scatter matrix using Plotly (requires Plotly installation).
        **plot_kws: Additional keyword arguments for the plot function.

    Returns:
        pd.DataFrame: The transformed data with PC columns.
        sns.PairGrid or plotly.graph_objs._figure.Figure: The plot object.
    """
    g = None
    df = df.copy()
    if scale:
        scaler = StandardScaler()
        df.loc[:, :] = scaler.fit_transform(df)
    pca = PCA(n_components)
    proj = pd.DataFrame(pca.fit_transform(df))
    proj.columns = proj.columns.values + 1
    proj = proj.add_prefix("PC-")
    proj.index = df.index
    if labels is not None:
        proj["label"] = list(labels)
    if plot:
        if not interactive:
            fig = sns.pairplot(proj, hue="label" if labels is not None else None, **plot_kws)
        else:
            pc_cols = proj.filter(regex='PC').columns.to_list()
            ndx_names = list(proj.index.names)
            fig = px.scatter_matrix(proj.reset_index(), color="label" if labels is not None else None,
                                    dimensions=pc_cols, hover_data=ndx_names, **plot_kws)
    return proj, fig


def quick_tsne(df, perplexity=30, metric='euclidean', plot=True, **kwargs):
    """
    Perform t-SNE dimensionality reduction on a DataFrame and optionally create a scatterplot.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data to be visualized.
    - perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 30.
    - metric (str, optional): The distance metric used in t-SNE. Defaults to 'euclidean'.
    - plot (bool, optional): If True, create a scatterplot. If False, return the reduced data. Defaults to True.
    - **kwargs: Additional keyword arguments to pass to the sns.scatterplot function.

    Returns:
    If plot is True, returns a tuple (df_tsne, ax), where df_tsne is the DataFrame containing the t-SNE results
    and ax is the scatterplot axes.
    If plot is False, returns only df_tsne.

    Example:
    df, ax = quick_tsne(my_data, perplexity=50, palette='viridis')
    plt.show()

    Notes:
    - This function uses t-SNE to reduce the dimensionality of the input DataFrame.
    - The resulting t-SNE coordinates are stored in a new DataFrame (df_tsne).
    - If plot is True, a scatterplot is created using seaborn.scatterplot, and the axes are returned along with df_tsne.
    - Additional keyword arguments (kwargs) can be provided to customize the scatterplot.

    """
    # Perform t-SNE dimensionality reduction
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='pca', perplexity=perplexity, metric=metric,
                      random_state=42).fit_transform(df.values)

    # Create a DataFrame with t-SNE results
    df_tsne = pd.DataFrame({'tSNE-1': X_embedded[:, 0], 'tSNE-2': X_embedded[:, 1]})

    if plot:
        # If plot is True, create a scatterplot
        ax = sns.scatterplot(data=df_tsne, x='tSNE-1', y='tSNE-2', **kwargs)
        ax.legend(bbox_to_anchor=(1, 1))

        return df_tsne, ax
    else:
        # If plot is False, return only df_tsne
        return df_tsne


def pycaret_score_threshold_analysis(pycaret_prediction):
    """
    Analyze the effect of different score thresholds on Balanced Accuracy and sample fractions.

    Args:
        pycaret_prediction: pd.DataFrame - The prediction DataFrame from PyCaret.

    Returns:
        None
    """
    score_thresholds = np.arange(0.5, 0.95, 0.01)
    accs = []
    ns = []

    for st in score_thresholds:
        tmp = pycaret_prediction[pycaret_prediction.Score > st]
        score = balanced_accuracy_score(tmp.DEATH_IND, tmp.Label)
        accs.append(score)
        ns.append(len(tmp) / len(pycaret_prediction))

    plot(score_thresholds, accs, color="C0")
    ylabel("Balanced accuracy", color="C0")
    xlabel("Score threshold")
    yticks(color="C0")

    ax1 = gca()
    ax2 = ax1.twinx()

    plot(score_thresholds, ns, color="C2")

    ylabel("Fraction of samples", color="C2")
    yticks(color="C2")
    grid()

    title("Score theshold analysis")


class ShapAnalysis:
    def __init__(self, model, df):
        """
        Initialize the ShapAnalysis class to perform SHAP value analysis on a given model and dataset.

        Args:
            model: object - The machine learning model for SHAP analysis.
            df: pd.DataFrame - The dataset for which SHAP values will be calculated.
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        self.df = df
        self.shap_values = shap_values
        self.instance_names = df.index.to_list()
        self.feature_names = df.columns.to_list()

    def waterfall(self, i, **kwargs):
        """
        Create a waterfall plot for a specific instance to explain its prediction.

        Args:
            i: int - Index of the instance to explain.
            **kwargs: Additional keyword arguments for the waterfall plot.
        """
        shap_values = self.shap_values
        self._base_values = shap_values[i][0].base_values
        self._values = shap_values[i].values
        shap_object = shap.Explanation(
            base_values=self._base_values,
            values=self._values,
            feature_names=self.feature_names,
            data=shap_values[i].data,
        )
        shap.plots.waterfall(shap_object, **kwargs)

    def summary(self, df=None, **kwargs):
        """
        Create a summary plot of SHAP values.

        Args:
            df: pd.DataFrame - The dataset for which the summary plot will be created (optional).
            **kwargs: Additional keyword arguments for the summary plot.
        """
        shap.summary_plot(self.shap_values, df if df is not None else self.df, **kwargs)

    def bar(self, **kwargs):
        """
        Create a bar plot to display feature importances based on SHAP values.

        Args:
            **kwargs: Additional keyword arguments for the bar plot.
        """
        shap.plots.bar(self.shap_values, **kwargs)
        for ax in plt.gcf().axes:
            for ch in ax.get_children():
                try:
                    ch.set_color("0.3")
                except:
                    break
