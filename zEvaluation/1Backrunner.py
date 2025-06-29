import subprocess
import json
import re
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import csv

class CommandRunner:
    def __init__(self, results_file: str = "command_results.json"):
        """
        Initialize the CommandRunner with a results file to store all command outputs.
        
        Args:
            results_file: Path to JSON file where results will be stored
        """
        self.results_file = results_file
        self.results = self.load_results()
    
    def load_results(self) -> List[Dict]:
        """Load existing results from file, or create empty list if file doesn't exist."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def save_results(self):
        """Save current results to file."""

        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def run_command(self, command: str, extract_patterns: Optional[Dict[str, str]] = None, 
                   timeout: int = 300, description: str = "") -> Dict[str, Any]:
        """
        Run a bash command and capture its output.
        
        Args:
            command: The bash command to execute
            extract_patterns: Dictionary of {variable_name: regex_pattern} to extract specific values
            timeout: Maximum time to wait for command completion (seconds)
            description: Human-readable description of what this command does
            
        Returns:
            Dictionary containing command info, output, extracted variables, and metadata
        """
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "description": description,
            "status": "running",
            "stdout": "",
            "stderr": "",
            "return_code": None,
            "extracted_variables": {},
            "execution_time": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Run the command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
                )
            
            # Wait for completion with timeout
            stdout, stderr = process.communicate(timeout=timeout)
            
            result["stdout"] = stdout
            result["stderr"] = stderr
            result["return_code"] = process.returncode
            result["status"] = "completed" if process.returncode == 0 else "failed"
            
        except subprocess.TimeoutExpired:
            process.kill()
            result["status"] = "timeout"
            result["stderr"] = f"Command timed out after {timeout} seconds"
            
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = str(e)
        
        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()
        
        # Extract variables from output if patterns provided
        if extract_patterns and result["stdout"]:
            result["extracted_variables"] = self.extract_variables(
                result["stdout"], extract_patterns
            )
        
        # Always extract APPLOG entries
        result["applog_entries"] = self.extract_applog_entries(result["stdout"])
        
        # Add to results list and save
        self.results.append(result)
        self.save_results()
        
        #print(f"Command completed with status: {result['status']}")
        
        return result
    
    def extract_applog_entries(self, text: str) -> List[str]:
        """
        Extract all APPLOG entries from the command output.
        
        Args:
            text: Text to search for APPLOG entries
            
        Returns:
            List of APPLOG entries found in the output
        """
        applog_pattern = r"APPLOG:\s*(.+)"
        matches = re.findall(applog_pattern, text, re.MULTILINE)
        return matches
    
    def extract_variables(self, text: str, patterns: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract variables from text using regex patterns.
        
        Args:
            text: Text to search in
            patterns: Dictionary of {variable_name: regex_pattern}
            
        Returns:
            Dictionary of extracted variables
        """
        extracted = {}
        #print("Extracting patterns" , patterns)
        for var_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                # If multiple matches, take the last one (often the final result)
                value = matches[-1]
                # Try to convert to number if possible
                if isinstance(value, str):
                    try:
                        if '.' in value:
                            extracted[var_name] = float(value)
                        else:
                            extracted[var_name] = int(value)
                    except ValueError:
                        extracted[var_name] = value
                else:
                    extracted[var_name] = value
        
        return extracted
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all command results."""
        if not self.results:
            return {"total_commands": 0, "summary": "No commands executed yet"}
        
        total = len(self.results)
        successful = len([r for r in self.results if r["status"] == "completed"])
        failed = len([r for r in self.results if r["status"] == "failed"])
        
        return {
            "total_commands": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total * 100 if total > 0 else 0,
            "latest_command": self.results[-1]["command"] if self.results else None
        }
    
    def export_to_csv(self, csv_file: str = "results_comparison.csv"):
        """Export extracted variables and APPLOG entries to CSV for easy comparison."""
        if not self.results:
            print("No results to export")
            return
        
        # Collect all unique variable names
        all_vars = set()
        for result in self.results:
            all_vars.update(result["extracted_variables"].keys())
        
        # Create CSV
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ["timestamp", "command", "description", "status", "execution_time", "applog_count", "applog_entries"] + list(all_vars)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    "timestamp": result["timestamp"],
                    "description": result["description"],
                    "status": result["status"],
                    "execution_time": result["execution_time"],
                    "applog_count": len(result.get("applog_entries", [])),
                    "applog_entries": " | ".join(result.get("applog_entries", []))
                }
                row.update(result["extracted_variables"])
                writer.writerow(row)
        with open("ml_model_results.json", "w") as f:
            f.write("")
        print(f"Results exported to {csv_file}")
    
    def get_applog_summary(self) -> Dict[str, Any]:
        """Get a summary of all APPLOG entries across all commands."""
        all_applogs = []
        for result in self.results:
            if "applog_entries" in result:
                all_applogs.extend(result["applog_entries"])
        
        return {
            "total_applog_entries": len(all_applogs),
            "unique_entries": len(set(all_applogs)),
            "all_entries": all_applogs
        }
    
    def ensemble_result(self):
        values = [
        entry["extracted_variables"]["annualized_volatility"]
        for entry in self.results
        if "extracted_variables" in entry and "annualized_volatility" in entry["extracted_variables"]
        ]
        # Compute the average
        average = sum(values) / len(values) if values else None
        return average
        

# Example usage and predefined patterns for common ML outputs
def main():
    # Initialize the runner
    runner = CommandRunner("ml_model_results.json")
    folder = r'sliced_data'
    start_date_str = "2024-01-12"  # <-- Set your manual start date here
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()

    for file in sorted(os.listdir(folder)):
        if file.endswith('.csv'):
            file_path = os.path.join(folder, file)
            date_str = os.path.splitext(file)[0]  # e.g., '2022-06-01' from '2022-06-01.csv'
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if file_date < start_date:
                continue
            print(f"Processing date: {date_str} in File: {file_path}")

            command1 = f"python zGen1\\RUNforecast.py --data {file_path}"
            result1 = runner.run_command(
                command1, 
                extract_patterns={"annualized_volatility": r"Annualized Volatility \(%\)[:\s]+([0-9.]+)"},
                description="zGen1"
            )
            command2 = ["zGen2\\.venv\\Scripts\\python.exe","zGen2\\main.py","predict","--data", f"{file_path}","--model", "models\\","--output", "forecasts\\"]  
            result2 = runner.run_command(
                command2, 
                extract_patterns={"annualized_volatility": r"Annualized Volatility\s*\(%\):\s*([\d\.]+)"},
                description="zGen2"
            )
            command3 = f"python zGen3\\NEXT.py predict --csv-path {file_path} --model lstm"
            result3 = runner.run_command(
                command3, 
                extract_patterns={"annualized_volatility": r"Annualized Volatility\s*\(%\):\s*([\d\.]+)"},
               description="zGen3lastm"
            )
            command4 = f"python zGen4\\rv_forecast.py --mode forecast --data_path {file_path}" ## Not working
            result4 = runner.run_command(
                command4, 
                extract_patterns={"annualized_volatility": r"Annualized Volatility \(%\)[:\s]+([0-9.]+)"},
                description="zGen4"
            )
            command5 = f"python zGen3\\NEXT.py predict --csv-path {file_path} --model transformer"
            result5 = runner.run_command(
                command5, 
                extract_patterns={"annualized_volatility": r"Annualized Volatility \(%\)[:\s]+([0-9.]+)"},
                description="zGen3tf"
            )
            """
            # Example: Another model for comparison
            command2 = "bash command"
            result2 = runner.run_command(
                command2, 
                extract_patterns={"variable to extract": r"variable to extract[:\s]+([0-9.-]+)"},
               description="description"
            )
            # """
            # Print summary
            ensemble = runner.ensemble_result()
            if ensemble is not None and isinstance(ensemble, (int, float)):
                print(f"RV for {date_str}: {ensemble:.6f} ")

            # Export to CSV for further analysis
            output_csv = f"slicerresults/{date_str}.csv"
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            if not os.path.exists(output_csv):
                with open(output_csv, "w") as f:
                    pass  # Create the file if it doesn't exist
            # Clear the JSON file completely
            with open("ml_model_results.json", "w") as f:
                f.write("[]")  
            runner.export_to_csv(output_csv)
            
            runner.results.clear()
if __name__ == "__main__":
    main()