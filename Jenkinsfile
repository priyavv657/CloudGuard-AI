
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install') {
            steps {
                bat '"C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" -m venv venv'
                bat 'venv\\Scripts\\python.exe -m pip install --upgrade pip'
                bat 'venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt'
            }
        }

        stage('Validate Data') {
            steps {
                bat 'venv\\Scripts\\python.exe src\\data_validation.py'
            }
        }

        stage('Test') {
            steps {
                bat 'venv\\Scripts\\python.exe -m pytest tests -v'
            }
        }

        stage('Train + MLflow') {
            steps {
                bat 'venv\\Scripts\\python.exe src\\train.py'
            }
        }

        stage('Archive Model') {
            steps {
                archiveArtifacts artifacts: 'models/*', fingerprint: true
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deployment is handled separately through Streamlit Community Cloud.'
            }
        }
    }

    post {
        success {
            echo 'Build SUCCESS'
        }
        failure {
            echo 'Build FAILED'
        }
    }
}
