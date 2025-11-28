import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import joblib
from pathlib import Path

# Bibliothèques de modélisation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InstagramEngagementPredictor:
    """
    Classe principale pour l'analyse et la prédiction du taux d'engagement Instagram.
    """
    
    def __init__(self, data_path='Instagram_Analytics.csv'):
        """
        Initialisation avec chemin vers le fichier de données.
        
        Args:
            data_path (str): Chemin vers le fichier CSV
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = None
        
    def load_data(self):
        """Chargement et exploration initiale des données."""
        print("="*60)
        print("1. CHARGEMENT DES DONNÉES")
        print("="*60)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Fichier {self.data_path} non trouvé!")
        
        # Chargement
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dimensions : {self.df.shape}")
        print(f"Colonnes : {list(self.df.columns)}")
        print("\nPremiers échantillons :")
        print(self.df.head())
        
        # Statistiques descriptives
        print("\nStatistiques descriptives :")
        print(self.df.describe())
        
        return self.df
    
    def preprocess_data(self):
        """Prétraitement complet des données."""
        print("\n" + "="*60)
        print("2. PRÉTRAITEMENT DES DONNÉES")
        print("="*60)
        
        if self.df is None:
            self.load_data()
        
        # 2.1 Conversion datetime et extraction temporelle
        print("Extraction des caractéristiques temporelles...")
        self.df['upload_date'] = pd.to_datetime(self.df['upload_date'])
        self.df['upload_year'] = self.df['upload_date'].dt.year
        self.df['upload_month'] = self.df['upload_date'].dt.month
        self.df['upload_day_of_week'] = self.df['upload_date'].dt.dayofweek
        self.df['upload_hour'] = self.df['upload_date'].dt.hour
        
        print(f"Nouvelles dimensions après features temporelles : {self.df.shape}")
        
        # 2.2 Encodage One-Hot
        print("Encodage des variables catégorielles...")
        categorical_cols = ['media_type', 'traffic_source', 'content_category']
        
        # Vérification que les colonnes existent
        missing_cols = [col for col in categorical_cols if col not in self.df.columns]
        if missing_cols:
            print(f"Attention : colonnes manquantes {missing_cols}")
        
        self.df_encoded = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        print(f"Dimensions après encodage One-Hot : {self.df_encoded.shape}")
        
        # 2.3 Nettoyage final
        print("Nettoyage final...")
        columns_to_drop = ['post_id', 'upload_date']
        available_cols_to_drop = [col for col in columns_to_drop if col in self.df_encoded.columns]
        
        self.df_processed = self.df_encoded.drop(columns=available_cols_to_drop)
        print(f"Dimensions finales : {self.df_processed.shape}")
        print(f"Colonnes finales : {len(self.df_processed.columns)}")
        
        return self.df_processed
    
    def check_missing_values(self):
        """Vérification des valeurs manquantes."""
        print("\n" + "="*60)
        print("3. VÉRIFICATION DES VALEURS MANQUANTES")
        print("="*60)
        
        if self.df_processed is None:
            self.preprocess_data()
        
        print("Valeurs manquantes par colonne :")
        missing_values = self.df_processed.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Statistiques sur la variable cible
        target_stats = self.df_processed['engagement_rate'].describe()
        print(f"\nDistribution de engagement_rate :")
        print(target_stats)
        
        total_missing = missing_values.sum()
        if total_missing == 0:
            print("✅ Aucune valeur manquante détectée!")
        else:
            print(f"⚠️  {total_missing} valeurs manquantes au total")
        
        return total_missing == 0
    
    def exploratory_data_analysis(self, save_plots=True):
        """Analyse exploratoire visuelle."""
        print("\n" + "="*60)
        print("4. ANALYSE EXPLORATOIRE VISUELLE (EDA)")
        print("="*60)
        
        if self.df_processed is None:
            self.preprocess_data()
        
        # Configuration des graphiques
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Analyse Exploratoire - Distributions Principales', fontsize=16, y=0.98)
        
        # 4.1 Distribution de la variable cible
        axes[0,0].hist(self.df_processed['engagement_rate'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution du Taux d\'Engagement')
        axes[0,0].set_xlabel('Taux d\'Engagement (%)')
        axes[0,0].set_ylabel('Fréquence')
        
        # 4.2 Boxplot likes vs engagement_rate (échantillon pour éviter surcharge)
        sample_data = self.df_processed.sample(min(1000, len(self.df_processed)))
        sns.boxplot(data=sample_data, x='likes', y='engagement_rate', ax=axes[0,1])
        axes[0,1].set_title('Likes vs Taux d\'Engagement (échantillon)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 4.3 Distribution des types de média (du df original)
        if 'media_type' in self.df.columns:
            media_counts = self.df['media_type'].value_counts()
            axes[0,2].pie(media_counts.values, labels=media_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0,2].set_title('Répartition par Type de Contenu')
        else:
            axes[0,2].text(0.5, 0.5, 'Données\ncatégorielles\nnon disponibles', 
                          ha='center', va='center', transform=axes[0,2].transAxes)
            axes[0,2].set_title('Types de Contenu')
        
        # 4.4 Heatmap de corrélation (variables numériques principales)
        numeric_cols = ['likes', 'comments', 'shares', 'saves', 'reach', 'impressions', 
                       'engagement_rate']
        available_numeric = [col for col in numeric_cols if col in self.df_processed.columns]
        
        if len(available_numeric) > 1:
            corr_matrix = self.df_processed[available_numeric].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', ax=axes[1,0], square=True)
            axes[1,0].set_title('Corrélations - Variables Numériques')
        else:
            axes[1,0].text(0.5, 0.5, 'Variables\nnumériques\ninsuffisantes', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Corrélations')
        
        # 4.5 Distribution horaire
        if 'upload_hour' in self.df_processed.columns:
            hourly_engagement = self.df_processed.groupby('upload_hour')['engagement_rate'].mean()
            axes[1,1].plot(hourly_engagement.index, hourly_engagement.values, marker='o', linewidth=2)
            axes[1,1].set_title('Engagement Moyen par Heure de Publication')
            axes[1,1].set_xlabel('Heure (24h)')
            axes[1,1].set_ylabel('Taux d\'Engagement Moyen')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Données\nhoraires\nnon disponibles', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Analyse Horaire')
        
        # 4.6 Distribution par jour de la semaine
        if 'upload_day_of_week' in self.df_processed.columns:
            days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
            weekly_engagement = self.df_processed.groupby('upload_day_of_week')['engagement_rate'].mean()
            axes[1,2].bar(range(7), weekly_engagement.values, color='lightcoral', alpha=0.7)
            axes[1,2].set_title('Engagement Moyen par Jour de la Semaine')
            axes[1,2].set_xlabel('Jour de la Semaine')
            axes[1,2].set_ylabel('Taux d\'Engagement Moyen')
            axes[1,2].set_xticks(range(7))
            axes[1,2].set_xticklabels(days)
            axes[1,2].grid(True, alpha=0.3, axis='y')
        else:
            axes[1,2].text(0.5, 0.5, 'Données\nhebdomadaires\nnon disponibles', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Analyse Hebdomadaire')
        
        plt.tight_layout()
        
        # Sauvegarde des graphiques
        if save_plots:
            output_dir = Path("output/eda_plots")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "eda_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Graphiques EDA sauvegardés dans {output_dir}")
        
        plt.show()
        
        # Statistiques supplémentaires
        print(f"\nCorrélation entre likes et engagement_rate : {self.df_processed['likes'].corr(self.df_processed['engagement_rate']):.4f}")
        print(f"Corrélation entre reach et engagement_rate : {self.df_processed['reach'].corr(self.df_processed['engagement_rate']):.4f}")
    
    def prepare_data_for_modeling(self):
        """Préparation des données pour la modélisation."""
        print("\n" + "="*60)
        print("5. PRÉPARATION DES DONNÉES POUR LA MODÉLISATION")
        print("="*60)
        
        if self.df_processed is None:
            self.preprocess_data()
        
        # Séparation features / cible
        if 'engagement_rate' not in self.df_processed.columns:
            raise ValueError("Colonne 'engagement_rate' non trouvée!")
        
        self.y = self.df_processed['engagement_rate']
        self.X = self.df_processed.drop(columns=['engagement_rate'])
        
        print(f"Nombre de features : {self.X.shape[1]}")
        print(f"Nombre d'observations : {self.X.shape[0]}")
        print(f"Variable cible - Moyenne : {self.y.mean():.2f}, Écart-type : {self.y.std():.2f}")
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Ensemble d'entraînement : {self.X_train.shape}")
        print(f"Ensemble de test : {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_linear_regression(self):
        """Entraînement du modèle de régression linéaire."""
        print("\n" + "-"*40)
        print("6.1 RÉGRESSION LINÉAIRE")
        print("-"*40)
        
        if self.X_train is None:
            self.prepare_data_for_modeling()
        
        # Modèle
        model_lr = LinearRegression()
        model_lr.fit(self.X_train, self.y_train)
        y_pred_lr = model_lr.predict(self.X_test)
        
        # Métriques
        mse_lr = mean_squared_error(self.y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)
        r2_lr = r2_score(self.y_test, y_pred_lr)
        
        self.models['linear_regression'] = {
            'model': model_lr,
            'predictions': y_pred_lr,
            'r2': r2_lr,
            'mse': mse_lr,
            'rmse': rmse_lr
        }
        
        print(f"R² : {r2_lr:.4f}")
        print(f"MSE : {mse_lr:.2f}")
        print(f"RMSE : {rmse_lr:.2f}")
        
        # Visualisation
        self._plot_predictions_vs_actual(self.y_test, y_pred_lr, "Régression Linéaire", "linear_regression")
        
        return model_lr, r2_lr, mse_lr, rmse_lr
    
    def train_polynomial_regression(self, degree=2):
        """Entraînement de la régression polynomiale."""
        print("\n" + "-"*40)
        print(f"6.2 RÉGRESSION POLYNOMIALE (degré {degree})")
        print("-"*40)
        
        # Transformation polynomiale
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(self.X_train)
        X_test_poly = poly_features.transform(self.X_test)
        
        print(f"Nombre de features après transformation polynomiale : {X_train_poly.shape[1]}")
        
        # Modèle
        model_poly = LinearRegression()
        model_poly.fit(X_train_poly, self.y_train)
        y_pred_poly = model_poly.predict(X_test_poly)
        
        # Métriques
        mse_poly = mean_squared_error(self.y_test, y_pred_poly)
        rmse_poly = np.sqrt(mse_poly)
        r2_poly = r2_score(self.y_test, y_pred_poly)
        
        self.models['polynomial_regression'] = {
            'model': model_poly,
            'poly_features': poly_features,
            'predictions': y_pred_poly,
            'r2': r2_poly,
            'mse': mse_poly,
            'rmse': rmse_poly
        }
        
        print(f"R² : {r2_poly:.4f}")
        print(f"MSE : {mse_poly:.2f}")
        print(f"RMSE : {rmse_poly:.2f}")
        
        # Visualisation
        self._plot_predictions_vs_actual(self.y_test, y_pred_poly, "Régression Polynomiale", "polynomial_regression")
        
        return model_poly, r2_poly, mse_poly, rmse_poly
    
    def train_decision_tree(self, max_depth=10):
        """Entraînement de l'arbre de décision."""
        print("\n" + "-"*40)
        print("6.3 ARBRE DE DÉCISION")
        print("-"*40)
        
        # Modèle
        model_dt = DecisionTreeRegressor(random_state=42, max_depth=max_depth)
        model_dt.fit(self.X_train, self.y_train)
        y_pred_dt = model_dt.predict(self.X_test)
        
        # Métriques
        mse_dt = mean_squared_error(self.y_test, y_pred_dt)
        rmse_dt = np.sqrt(mse_dt)
        r2_dt = r2_score(self.y_test, y_pred_dt)
        
        self.models['decision_tree'] = {
            'model': model_dt,
            'predictions': y_pred_dt,
            'r2': r2_dt,
            'mse': mse_dt,
            'rmse': rmse_dt
        }
        
        print(f"R² : {r2_dt:.4f}")
        print(f"MSE : {mse_dt:.2f}")
        print(f"RMSE : {rmse_dt:.2f}")
        
        # Analyse d'importance des features
        self._analyze_feature_importance(model_dt, "decision_tree")
        
        # Visualisation
        self._plot_predictions_vs_actual(self.y_test, y_pred_dt, "Arbre de Décision", "decision_tree")
        
        return model_dt, r2_dt, mse_dt, rmse_dt
    
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """Entraînement de la forêt aléatoire."""
        print("\n" + "-"*40)
        print("6.4 FORÊT ALÉATOIRE")
        print("-"*40)
        
        # Modèle
        model_rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42, 
            max_depth=max_depth,
            n_jobs=-1
        )
        model_rf.fit(self.X_train, self.y_train)
        y_pred_rf = model_rf.predict(self.X_test)
        
        # Métriques
        mse_rf = mean_squared_error(self.y_test, y_pred_rf)
        rmse_rf = np.sqrt(mse_rf)
        r2_rf = r2_score(self.y_test, y_pred_rf)
        
        self.models['random_forest'] = {
            'model': model_rf,
            'predictions': y_pred_rf,
            'r2': r2_rf,
            'mse': mse_rf,
            'rmse': rmse_rf
        }
        
        print(f"R² : {r2_rf:.4f}")
        print(f"MSE : {mse_rf:.2f}")
        print(f"RMSE : {rmse_rf:.2f}")
        
        # Analyse d'importance des features
        self._analyze_feature_importance(model_rf, "random_forest")
        
        # Visualisation
        self._plot_predictions_vs_actual(self.y_test, y_pred_rf, "Forêt Aléatoire", "random_forest")
        
        return model_rf, r2_rf, mse_rf, rmse_rf
    
    def train_svr(self):
        """Entraînement du SVR."""
        print("\n" + "-"*40)
        print("6.5 SVR (Support Vector Regression)")
        print("-"*40)
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Modèle SVR
        model_svr = SVR(kernel='rbf', C=1.0, gamma='scale')
        model_svr.fit(X_train_scaled, self.y_train)
        y_pred_svr = model_svr.predict(X_test_scaled)
        
        # Métriques
        mse_svr = mean_squared_error(self.y_test, y_pred_svr)
        rmse_svr = np.sqrt(mse_svr)
        r2_svr = r2_score(self.y_test, y_pred_svr)
        
        self.models['svr'] = {
            'model': model_svr,
            'scaler': scaler,
            'predictions': y_pred_svr,
            'r2': r2_svr,
            'mse': mse_svr,
            'rmse': rmse_svr
        }
        
        print(f"R² : {r2_svr:.4f}")
        print(f"MSE : {mse_svr:.2f}")
        print(f"RMSE : {rmse_svr:.2f}")
        
        # Visualisation
        self._plot_predictions_vs_actual(self.y_test, y_pred_svr, "SVR", "svr")
        
        return model_svr, r2_svr, mse_svr, rmse_svr
    
    def train_all_models(self):
        """Entraînement de tous les modèles."""
        print("\n" + "="*60)
        print("6. ENTRAÎNEMENT DE TOUS LES MODÈLES")
        print("="*60)
        
        # Entraînement séquentiel
        models_results = {}
        
        # Régression Linéaire
        lr_model, lr_r2, lr_mse, lr_rmse = self.train_linear_regression()
        models_results['Régression Linéaire'] = {'r2': lr_r2, 'mse': lr_mse, 'rmse': lr_rmse}
        
        # Régression Polynomiale
        poly_model, poly_r2, poly_mse, poly_rmse = self.train_polynomial_regression()
        models_results['Régression Polynomiale'] = {'r2': poly_r2, 'mse': poly_mse, 'rmse': poly_rmse}
        
        # Arbre de Décision
        dt_model, dt_r2, dt_mse, dt_rmse = self.train_decision_tree()
        models_results['Arbre de Décision'] = {'r2': dt_r2, 'mse': dt_mse, 'rmse': dt_rmse}
        
        # Forêt Aléatoire
        rf_model, rf_r2, rf_mse, rf_rmse = self.train_random_forest()
        models_results['Forêt Aléatoire'] = {'r2': rf_r2, 'mse': rf_mse, 'rmse': rf_rmse}
        
        # SVR
        svr_model, svr_r2, svr_mse, svr_rmse = self.train_svr()
        models_results['SVR'] = {'r2': svr_r2, 'mse': svr_mse, 'rmse': svr_rmse}
        
        # Création du DataFrame des résultats
        self.results = pd.DataFrame(models_results).T.round(4)
        print("\n" + "="*60)
        print("RÉSULTATS COMPARATIFS")
        print("="*60)
        print(self.results)
        
        return self.results
    
    def compare_models(self, save_plots=True):
        """Comparaison visuelle de tous les modèles."""
        print("\n" + "="*60)
        print("7. COMPARAISON VISUELLE DES MODÈLES")
        print("="*60)
        
        if self.results is None:
            self.train_all_models()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        models = self.results.index
        r2_scores = self.results['r2']
        rmse_scores = self.results['rmse']
        
        # 7.1 Comparaison R²
        colors = ['red' if score < 0.2 else 'orange' if score < 0.5 else 'green' 
                 if score < 0.7 else 'darkgreen' for score in r2_scores]
        
        bars1 = ax1.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Comparaison des Scores R² par Modèle', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score R²')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Ajout des valeurs sur les barres
        for bar, score in zip(bars1, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 7.2 Comparaison RMSE
        bars2 = ax2.bar(models, rmse_scores, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Comparaison des Erreurs RMSE par Modèle', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Ajout des valeurs sur les barres
        for bar, score in zip(bars2, rmse_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.1f}', ha='center', va='bottom', rotation=45)
        
        # 7.3 Classement des modèles par R²
        ranking = self.results.sort_values('r2', ascending=False)
        ax3.barh(range(len(ranking)), ranking['r2'], color=colors[::-1], alpha=0.7)
        ax3.set_yticks(range(len(ranking)))
        ax3.set_yticklabels(ranking.index)
        ax3.set_xlabel('Score R²')
        ax3.set_title('Classement des Modèles (par R²)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Ajout des valeurs
        for i, (idx, row) in enumerate(ranking.iterrows()):
            ax3.text(row['r2'] + 0.01, i, f'{row["r2"]:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Sauvegarde
        if save_plots:
            output_dir = Path("output/model_comparison")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Graphiques de comparaison sauvegardés dans {output_dir}")
        
        plt.show()
        
        # Identification du meilleur modèle
        best_idx = self.results['r2'].idxmax()
        best_r2 = self.results.loc[best_idx, 'r2']
        print(f"\n🏆 MEILLEUR MODÈLE : {best_idx}")
        print(f"   Score R² : {best_r2:.4f}")
        print(f"   RMSE : {self.results.loc[best_idx, 'rmse']:.2f}")
        
        return best_idx
    
    def _plot_predictions_vs_actual(self, y_true, y_pred, model_name, save_name):
        """Visualisation prédictions vs valeurs réelles."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, s=20, color='steelblue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label="Ligne parfaite")
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title(f'{model_name}: Prédictions vs Valeurs Réelles')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ajout du coefficient de corrélation
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        plt.text(0.05, 0.95, f'Corrélation: {corr:.3f}\nR²: {r2_score(y_true, y_pred):.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarde
        output_dir = Path("output/model_predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{save_name}_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_feature_importance(self, model, model_name):
        """Analyse de l'importance des features."""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Top 15 features
            top_features = importance_df.head(15)
            
            # Graphique
            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_features, x='importance', y='feature', 
                       palette='viridis', edgecolor='black')
            plt.title(f'Top 15 Features Importantes - {model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Importance Relative')
            plt.tight_layout()
            
            # Sauvegarde
            output_dir = Path("output/feature_importance")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"{model_name}_feature_importance.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nTop 10 features pour {model_name}:")
            for i, row in top_features.head(10).iterrows():
                print(f"   {i+1:2d}. {row['feature']:<30} : {row['importance']:.4f}")
            
            # Sauvegarde CSV
            top_features.to_csv(output_dir / f"{model_name}_feature_importance.csv", index=False)
    
    def save_results(self):
        """Sauvegarde complète des résultats."""
        print("\n" + "="*60)
        print("8. SAUVEGARDE DES RÉSULTATS")
        print("="*60)
        
        if self.results is None:
            self.train_all_models()
        
        output_dir = Path("output/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde DataFrame des résultats
        self.results.to_csv(output_dir / "model_performance_comparison.csv")
        
        # Sauvegarde du meilleur modèle
        best_model_name = self.results['r2'].idxmax()
        best_model_key = best_model_name.lower().replace(' ', '_').replace('é', 'e')
        
        if best_model_key in self.models:
            joblib.dump(self.models[best_model_key]['model'], 
                       output_dir / f"best_model_{best_model_key}.pkl")
            
            # Sauvegarde des métriques du meilleur modèle
            best_results = self.results.loc[best_model_name]
            best_results.to_dict()
            pd.Series(best_results).to_json(output_dir / "best_model_metrics.json")
        
        print(f"Résultats sauvegardés dans {output_dir}")
        print(f"Meilleur modèle sauvegardé : {best_model_name}")
    
    def generate_report(self):
        """Génération d'un rapport résumé."""
        print("\n" + "="*60)
        print("9. RAPPORT FINAL")
        print("="*60)
        
        if self.results is None:
            self.train_all_models()
        
        print("\n📊 RÉSUMÉ DES PERFORMANCES")
        print("-" * 50)
        print(self.results.to_string())
        
        best_model = self.results['r2'].idxmax()
        best_r2 = self.results.loc[best_model, 'r2']
        
        print(f"\n🏆 CONCLUSION")
        print(f"Le modèle {best_model} est le plus performant avec un R² de {best_r2:.3f}")
        print(f"   Cela signifie qu'il explique {best_r2*100:.1f}% de la variance du taux d'engagement.")
        print(f"   Erreur moyenne (RMSE) : {self.results.loc[best_model, 'rmse']:.2f}%")
        
        print(f"\n💡 RECOMMANDATIONS")
        print("1. Optimiser les hyperparamètres du meilleur modèle")
        print("2. Ajouter du feature engineering (ratios, interactions)")
        print("3. Tester des modèles d'ensemble avancés (XGBoost, LightGBM)")
        print("4. Implémenter une validation croisée")
        print("5. Analyser l'impact business des prédictions")
        
        return {
            'best_model': best_model,
            'best_r2': best_r2,
            'results': self.results
        }
    
    def run_complete_analysis(self, save_plots=True):
        """
        Exécution complète de l'analyse.
        
        Args:
            save_plots (bool): Sauvegarder les graphiques
        """
        print("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE")
        print("="*60)
        
        # Étape 1: Chargement
        self.load_data()
        
        # Étape 2: Prétraitement
        self.preprocess_data()
        
        # Étape 3: Vérification
        self.check_missing_values()
        
        # Étape 4: EDA
        self.exploratory_data_analysis(save_plots=save_plots)
        
        # Étape 5: Préparation
        self.prepare_data_for_modeling()
        
        # Étape 6: Entraînement
        self.train_all_models()
        
        # Étape 7: Comparaison
        best_model = self.compare_models(save_plots=save_plots)
        
        # Étape 8: Sauvegarde
        self.save_results()
        
        # Étape 9: Rapport
        report = self.generate_report()
        
        print("\n✅ ANALYSE TERMINÉE!")
        print(f"📁 Résultats sauvegardés dans le dossier 'output/'")
        
        return report


def main():
    """Fonction principale."""
    # Configuration du chemin du fichier
    DATA_FILE = "Instagram_Analytics.csv"
    
    try:
        # Création de l'analyseur
        analyzer = InstagramEngagementPredictor(data_path=DATA_FILE)
        
        # Exécution complète
        report = analyzer.run_complete_analysis(save_plots=True)
        
        # Affichage final
        print(f"\n🎯 RÉSULTAT FINAL")
        print(f"Meilleur modèle : {report['best_model']}")
        print(f"Performance R² : {report['best_r2']:.4f}")
        
    except FileNotFoundError as e:
        print(f"ERREUR : {e}")
        print(f"Vérifiez que le fichier '{DATA_FILE}' est dans le répertoire courant.")
        print("\nPour tester avec un fichier exemple, téléchargez 'Instagram_Analytics.csv' depuis Kaggle.")
    
    except Exception as e:
        print(f"ERREUR INATTENDUE : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

