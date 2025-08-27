module "serverless-streamlit-app" {
  source = "aws-ia/serverless-streamlit-app/aws"
  
  app_name        = "equitylens-app"
  app_version     = "v1.0.0"
  path_to_app_dir = "./app"
  
  # Optional: Configure resources
  task_cpu    = 512
  task_memory = 1024
  
  # Tags for resource management
  tags = {
    Environment = "production"
    Project     = "EquityLens"
  }
}