import matplotlib.pyplot as plt
import seaborn as sns

def plot_age_distribution(data):
    """Plot of age distribution"""
    plt.hist(data['Age'].dropna(), bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Age Count")
    plt.show()

def plot_sex_distribution(data):
    """Plot of sex distribution"""
    plt.pie(data['Sex'].dropna(), bins=20)
    plt.title("Sex")
    plt.show()