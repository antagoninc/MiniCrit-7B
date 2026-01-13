# ================================================================
# MiniCrit Terraform Variables
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "minicrit"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "minicrit-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

# ================================================================
# VPC Configuration
# ================================================================

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

# ================================================================
# Node Configuration
# ================================================================

variable "cpu_instance_types" {
  description = "CPU node instance types"
  type        = list(string)
  default     = ["m6i.xlarge", "m6i.2xlarge"]
}

variable "cpu_node_min" {
  description = "Minimum CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_node_max" {
  description = "Maximum CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_node_desired" {
  description = "Desired CPU nodes"
  type        = number
  default     = 2
}

variable "gpu_instance_types" {
  description = "GPU node instance types"
  type        = list(string)
  default     = ["g5.xlarge", "g5.2xlarge"]
}

variable "gpu_node_min" {
  description = "Minimum GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max" {
  description = "Maximum GPU nodes"
  type        = number
  default     = 5
}

variable "gpu_node_desired" {
  description = "Desired GPU nodes"
  type        = number
  default     = 1
}
