pipeline {
    agent any
    stages {
        stage('Checkout')  { steps { checkout scm } }
        stage('Install')   { steps { bat 'python -m venv venv && venv\\Scripts\\pip install -r requirements-dev.txt' } }
        stage('Validate Data') { steps { bat 'venv\\Scripts\\python src\\data_validation.py' } }
        stage('Test')      { steps { bat 'venv\\Scripts\\python -m pytest tests -v' } }
        stage('Train + MLflow') { steps { bat 'venv\\Scripts\\python src\\train.py' } }
        stage('Archive Model') { steps { archiveArtifacts artifacts: 'models/*', fingerprint: true } }
        stage('Deploy') { steps { echo 'Merge to main -> Streamlit Community Cloud redeploys automatically' } }
    }
     post {
        success {
            echo 'Build SUCCESS'
        }
        failure {
            echo 'Build FAILED'
        }
    }
