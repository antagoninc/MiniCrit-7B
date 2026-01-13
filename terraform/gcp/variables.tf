# ================================================================
# MiniCrit GCP Terraform Variables
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
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
  description = "GKE cluster name"
  type        = string
  default     = "minicrit-cluster"
}

# ================================================================
# Network Configuration
# ================================================================

variable "subnet_cidr" {
  description = "Subnet CIDR"
  type        = string
  default     = "10.0.0.0/20"
}

variable "pods_cidr" {
  description = "Pods CIDR"
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_cidr" {
  description = "Services CIDR"
  type        = string
  default     = "10.8.0.0/20"
}

variable "master_cidr" {
  description = "Master CIDR"
  type        = string
  default     = "172.16.0.0/28"
}

variable "node_zones" {
  description = "Node zones"
  type        = list(string)
  default     = ["us-central1-a", "us-central1-b", "us-central1-c"]
}

# ================================================================
# Node Configuration
# ================================================================

variable "cpu_machine_type" {
  description = "CPU node machine type"
  type        = string
  default     = "n2-standard-4"
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

variable "gpu_machine_type" {
  description = "GPU node machine type"
  type        = string
  default     = "n1-standard-8"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-tesla-t4"
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
