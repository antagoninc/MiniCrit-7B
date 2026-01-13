# ================================================================
# MiniCrit Azure Terraform Variables
# Antagon Inc. | CAGE: 17E75 | UEI: KBSGT7CZ4AH3
# ================================================================

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
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
  description = "AKS cluster name"
  type        = string
  default     = "minicrit-aks"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.29"
}

# ================================================================
# Network Configuration
# ================================================================

variable "vnet_cidr" {
  description = "VNet CIDR"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "AKS subnet CIDR"
  type        = string
  default     = "10.0.0.0/20"
}

variable "private_endpoint_cidr" {
  description = "Private endpoint subnet CIDR"
  type        = string
  default     = "10.0.16.0/24"
}

# ================================================================
# Node Configuration
# ================================================================

variable "system_node_count" {
  description = "System node count"
  type        = number
  default     = 2
}

variable "system_vm_size" {
  description = "System node VM size"
  type        = string
  default     = "Standard_D2s_v3"
}

variable "cpu_vm_size" {
  description = "CPU node VM size"
  type        = string
  default     = "Standard_D4s_v3"
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

variable "gpu_vm_size" {
  description = "GPU node VM size"
  type        = string
  default     = "Standard_NC6s_v3"
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
