// Jenkins pipeline: lint and test the project inside its Docker image,
// publish JUnit results and notify Slack when a webhook is configured.
pipeline {
    agent any

    environment {
        IMAGE = "nyt-sentiment-ci:${env.BUILD_NUMBER}"
    }

    options {
        timeout(time: 20, unit: 'MINUTES')
    }

    stages {
        stage('Build image') {
            steps {
                sh 'docker build -t "$IMAGE" .'
            }
        }

        stage('Lint') {
            steps {
                sh 'docker run --rm "$IMAGE" ruff check src tests airflow'
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    mkdir -p reports
                    docker run --rm -v "$PWD/reports:/app/reports" "$IMAGE" \
                        python -m pytest --junitxml=reports/junit.xml
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/junit.xml'
            sh 'docker image rm -f "$IMAGE" || true'
        }
        success {
            script { notifySlack("✅ Build succeeded: ${env.JOB_NAME} #${env.BUILD_NUMBER}") }
        }
        failure {
            script { notifySlack("❌ Build failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}") }
        }
    }
}

def notifySlack(String message) {
    if (env.SLACK_WEBHOOK) {
        sh """curl -s -X POST -H 'Content-type: application/json' \
            --data '{"text":"${message}"}' "\$SLACK_WEBHOOK" """
    }
}
