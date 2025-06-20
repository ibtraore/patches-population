import gradio as gr
import warnings
import asyncio
import os
import json
import queue
import time
import sys
import io
import threading
from datetime import datetime
import shutil
from typing import List, Tuple
import base64
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import signal
import multiprocessing as mp
import pandas as pd
from src.ontology_population_project.crew.profile_crew.profile_crew import ProfileCrew
from src.ontology_population_project.crew.modules_crew.module_crew import ModuleCrew
from src.ontology_population_project.tools.aggregate_json import aggregate_triplets_json
from src.ontology_population_project.evaluation.evaluation import evaluation
from crewai.telemetry import Telemetry
from crewai.flow.flow import Flow, listen, start, and_ 
import nest_asyncio
from dotenv import load_dotenv
load_dotenv()

# Docker asyncio Configuration - COMPLETE SOLUTION
nest_asyncio.apply()

class DockerAsyncioManager:
    """Asyncio manager specifically designed for Docker"""
    
    @staticmethod
    def setup_event_loop() -> asyncio.AbstractEventLoop:
        """Configure an event loop for the current thread"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop closed")
        except RuntimeError:
            # Create a new loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    @staticmethod
    def patch_threading():
        """Patch all threads to have an event loop"""
        original_thread_run = threading.Thread.run
        
        def patched_run(self):
            # Ensure we have an event loop in each thread
            DockerAsyncioManager.setup_event_loop()
            try:
                return original_thread_run(self)
            except Exception as e:
                print(f"Thread error {self.name}: {e}")
                raise
        
        threading.Thread.run = patched_run
    
    @staticmethod
    def patch_uvicorn():
        """Specific patch for Uvicorn in Docker"""
        try:
            import uvicorn.server
            original_run = uvicorn.server.Server.run
            
            def patched_uvicorn_run(self, sockets=None):
                # Configure the event loop before starting
                loop = DockerAsyncioManager.setup_event_loop()
                if not loop.is_running():
                    return loop.run_until_complete(self.serve(sockets=sockets))
                else:
                    # If the loop is already running, create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(self.serve(sockets=sockets))
                        )
                        return future.result()
            
            uvicorn.server.Server.run = patched_uvicorn_run
        except ImportError:
            pass  # Uvicorn not installed

# Docker manager initialization
docker_manager = DockerAsyncioManager()
docker_manager.setup_event_loop()
docker_manager.patch_threading()
docker_manager.patch_uvicorn()

# Secure launch function for Docker
def launch_gradio_docker(demo, port=7860, host="0.0.0.0"):
    """Launch Gradio securely in Docker"""
    
    def find_available_port(start_port, max_attempts=10):
        """Find an available port"""
        import socket
        for i in range(max_attempts):
            test_port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((host, test_port))
                    return test_port
            except OSError:
                continue
        raise OSError(f"No available port in range {start_port}-{start_port + max_attempts}")
    
    # Find an available port
    try:
        available_port = find_available_port(port)
        print(f"🚀 Launching Gradio on http://localhost:7860 (bound to {host})")
      
        # Special configuration for Docker
        demo.launch(
            server_name=host,
            server_port=available_port,
            share=False,
            show_error=True,
            quiet=False,
            debug=False,
            # Docker-specific parameters
            max_threads=4,  # Limit the number of threads
        )
        
    except Exception as e:
        print(f"❌ Error during launch: {e}")
        # Fallback: try without threading
        try:
            print("🔄 Attempting without threading...")
            demo.launch(
                server_name=host,
                server_port=find_available_port(port + 10),
                share=False,
                show_error=True,
                quiet=False,
                debug=False,
            )
        except Exception as fallback_error:
            print(f"❌ Fallback failed: {fallback_error}")
            raise

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Disable telemetry
def noop(*args, **kwargs):
    pass

for attr in dir(Telemetry):
    if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
        setattr(Telemetry, attr, noop)

# Path configuration
GRADIO_OUTPUT_DIR = "src/ontology_population_project/agent-output/Gradio"
os.makedirs(GRADIO_OUTPUT_DIR, exist_ok=True)

class RealTimeLogCapture:
    """Class for capturing logs in real-time"""
    def __init__(self):
        self.logs = []
        self.log_queue = queue.Queue()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.capturing = False
        self.buffer = io.StringIO()
        
    def start_capture(self):
        """Start log capture"""
        self.capturing = True
        self.logs.clear()
        
        # Redirect stdout and stderr to our capture
        sys.stdout = self
        sys.stderr = self
        
    def stop_capture(self):
        """Stop log capture"""
        self.capturing = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
    def write(self, text):
        """Method called for each print/log"""
        # Write to original output to keep terminal display
        self.original_stdout.write(text)
        self.original_stdout.flush()
        
        if self.capturing and text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Clean text from ANSI special characters (terminal colors)
            import re
            clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text.strip())
            
            if clean_text:
                log_entry = f"[{timestamp}] {clean_text}"
                self.logs.append(log_entry)
                self.log_queue.put(log_entry)
        
        return len(text)
    
    def flush(self):
        """Flush required for stdout interface"""
        if hasattr(self.original_stdout, 'flush'):
            self.original_stdout.flush()
    
    def get_logs(self):
        """Retrieve all accumulated logs"""
        return "\n".join(self.logs)
    
    def get_new_logs(self):
        """Retrieve new logs since last call"""
        new_logs = []
        try:
            while True:
                log = self.log_queue.get_nowait()
                new_logs.append(log)
        except queue.Empty:
            pass
        return "\n".join(new_logs) if new_logs else ""

# Global instance of the log capture
log_capture = RealTimeLogCapture()

# Global variable to control log updates
pipeline_running = False

class GradioOntologyFlow(Flow):
    def __init__(self, pdf_path, patient_name, planning_data, output_dir):
        super().__init__()
        self.pdf_path = pdf_path
        self.patient_name = patient_name
        self.planning_data = planning_data
        self.output_dir = output_dir
        self.start_time = time.time()
    
    @start()
    async def profile_crew(self):
        print(f"🚀 Starting Profile for patient: {self.patient_name}")
        
        inputs_profile = {
            "pdf_path": self.pdf_path,
            "patient_name": self.patient_name
        }
        
        # Create a Profile instance configured for this patient
        profile_crew_instance = ProfileCrew(patient_name=self.patient_name)
        
        print("📋 Exécution de Profile en cours...")
        result = await profile_crew_instance.crew().kickoff_async(inputs=inputs_profile)
        print("✅ Profile terminé avec succès!")
        
        return result
    
    @listen(profile_crew)
    async def module_crew(self, context):
        print(f"🔧 Démarrage Module pour le patient: {self.patient_name}")
        
        # Inputs for ModuleCrew - uses JSON planning data
        inputs_module = {
            "module_text": self.planning_data,
        }
        
        # Create a ModuleCrew instance configured for this patient
        module_crew_instance = ModuleCrew(patient_name=self.patient_name)
        
        print("📊 Exécution de Module en cours...")
        result = await module_crew_instance.crew().kickoff_async(inputs=inputs_module)
        print("✅ Module terminé avec succès!")
        
        return result


    # Aggregate module and profile outputs
    @listen(and_(profile_crew, module_crew))
    async def final_aggregation(self, context):
        print(f"🔄 Démarrage de l'agrégation finale pour le patient: {self.patient_name}")
        
        print("📊 Exécution de l'agrégation JSON...")
        aggregate_triplets_json(self.output_dir)
        print("✅ Agrégation terminée avec succès!")
        
        return "Aggregation completed"

def create_pdf_viewer_html(pdf_path: str) -> str:
    """Create HTML viewer to display PDF in the interface"""
    if not pdf_path or not os.path.exists(pdf_path):
        return "<div style='text-align: center; padding: 20px; color: #666;'>No PDF to display</div>"
    
    try:
        # Convert PDF to base64 to embed in HTML
        with open(pdf_path, 'rb') as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Create HTML with embedded PDF viewer
        html_content = f"""
        <div style="width: 100%; height: 600px; border: 1px solid #ddd; border-radius: 8px;">
            <embed src="data:application/pdf;base64,{pdf_base64}" 
                   width="100%" 
                   height="100%" 
                   type="application/pdf"
                   style="border: none;">
            <div style="text-align: center; padding: 20px;">
                <p>Your browser does not support embedded PDF display.</p>
                <a href="data:application/pdf;base64,{pdf_base64}" 
                   download="{os.path.basename(pdf_path)}" 
                   style="color: #007bff; text-decoration: underline;">
                   Download PDF
                </a>
            </div>
        </div>
        """
        
        return html_content
        
    except Exception as e:
        return f"<div style='text-align: center; padding: 20px; color: #dc3545;'>Error loading PDF: {str(e)}</div>"

def get_patient_files(patient_dir: str) -> Tuple[str, str, str, str]:
    """Retrieves the contents of the 4 main files for a patient"""
    
    # 1. PDF document for display
    pdf_files = [f for f in os.listdir(patient_dir) if f.endswith('.pdf')]
    pdf_html = ""
    if pdf_files:
        pdf_path = os.path.join(patient_dir, pdf_files[0])
        pdf_html = create_pdf_viewer_html(pdf_path)
    else:
        pdf_html = "<div style='text-align: center; padding: 20px; color: #666;'>No PDF found</div>"
    
    # 2. Parsing result (llama_parser_output.txt)
    llama_parser_path = os.path.join(patient_dir, "llama_parser_output.txt")
    llama_content = load_file_safely(llama_parser_path)
    
    # 3. Profile Agent 2 (interpretation)
    agent2_path = os.path.join(patient_dir, "profile_interpreting.txt")
    agent2_content = load_file_safely(agent2_path)
    
    # 4. Profile Agent 4 (final JSON result)
    agent4_path = os.path.join(patient_dir, "profile_extraction.json")
    agent4_content = load_file_safely(agent4_path)
    
    return pdf_html, llama_content, agent2_content, agent4_content

def get_module_files(patient_dir: str) -> Tuple[str, str, str, str, str, str]:
    """Retrieves the contents of the 6 Module files for a patient"""
    
    # The 6 Module files (all in JSON)
    module_files = [
        "module_extraction_time.json",
        "module_extraction_situation.json", 
        "module_extraction_person.json",
        "module_extraction_environnement.json",
        "module_extraction_challenge.json",
        "module_extraction_activity.json"
    ]
    
    contents = []
    for filename in module_files:
        file_path = os.path.join(patient_dir, filename)
        content = load_file_safely(file_path)
        contents.append(content)
    
    return tuple(contents)

def load_file_safely(file_path: str) -> str:
    """Loads file content safely"""
    if not os.path.exists(file_path):
        return f"File not found: {os.path.basename(file_path)}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Format JSON if it's a JSON file
        if file_path.endswith('.json'):
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except:
                return content
        
        return content
    
    except Exception as e:
        return f"Error loading {os.path.basename(file_path)}: {str(e)}"

def list_processed_patients() -> List[str]:
    """Lists all processed patients"""
    patients = []
    if os.path.exists(GRADIO_OUTPUT_DIR):
        for item in os.listdir(GRADIO_OUTPUT_DIR):
            patient_dir = os.path.join(GRADIO_OUTPUT_DIR, item)
            if os.path.isdir(patient_dir) and item != "ground_truth":
                patients.append(item)
    return sorted(patients)

def validate_json_data(json_text):
    """Validates and formats JSON data"""
    try:
        data = json.loads(json_text)
        return True, json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return False, f"JSON Error: {str(e)}"

def create_pdf_preview(pdf_file):
    """Create preview of uploaded PDF"""
    if not pdf_file:
        return "<div style='text-align: center; padding: 20px; color: #666;'>No file selected</div>"
    
    try:
        # If it's a Gradio file object
        if hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            return "<div style='text-align: center; padding: 20px; color: #666;'>Unsupported file format</div>"
        
        return create_pdf_viewer_html(pdf_path)
        
    except Exception as e:
        return f"<div style='text-align: center; padding: 20px; color: #dc3545;'>Error: {str(e)}</div>"

def update_logs_realtime():
    """Updates logs in real-time during execution"""
    global pipeline_running
    if pipeline_running:
        new_logs = log_capture.get_new_logs()
        all_logs = log_capture.get_logs()
        return all_logs
    else:
        return log_capture.get_logs()

def run_pipeline_async(pdf_file, patient_name, planning_json):
    """Asynchronous version that executes the pipeline in a separate thread"""
    global pipeline_running
    
    if not pdf_file or not patient_name.strip() or not planning_json.strip():
        return "❌ Error: Please provide all required elements"
    
    # Validate JSON
    is_valid, formatted_json = validate_json_data(planning_json)
    if not is_valid:
        return f"❌ {formatted_json}"
    
    pipeline_running = True
    log_capture.start_capture()
    
    try:
        print("🔄 Pipeline initialization...")
        
        # Create directory for this patient
        patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Save PDF file
        print("💾 Saving PDF file...")
        pdf_path = os.path.join(patient_dir, f"criteria_{patient_name}.pdf")
        
        # Handle PDF file upload
        if hasattr(pdf_file, 'name'):
            shutil.copy2(pdf_file.name, pdf_path)
        else:
            with open(pdf_path, "wb") as f:
                f.write(pdf_file)
        
        print(f"🚀 Starting pipeline for patient: {patient_name}")
        print(f"📄 PDF file: {pdf_path}")
        print(f"📁 Output directory: {patient_dir}")
        
        # Create and launch flow
        flow = GradioOntologyFlow(
            pdf_path=pdf_path,
            patient_name=patient_name,
            planning_data=formatted_json,
            output_dir=patient_dir
        )
        
        # Execute flow asynchronously
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(flow.kickoff_async())
            status = "✅ Pipeline executed successfully!"
            print("🎉 Pipeline completed successfully!")
        except Exception as e:
            status = f"❌ Error during execution: {str(e)}"
            print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        status = f"❌ Error: {str(e)}"
        print(f"❌ General error: {str(e)}")
    
    finally:
        pipeline_running = False
        log_capture.stop_capture()
    
    return status

def run_pipeline_with_realtime_updates(pdf_file, patient_name, planning_excel, planning_json_preview):
    """Launch the pipeline with real-time updates"""

    def pipeline_thread():
        return run_pipeline_async(pdf_file, patient_name, planning_json_preview)    
    # Launch the pipeline in a separate thread
    thread = threading.Thread(target=pipeline_thread)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to finish while updating the interface
    while thread.is_alive():
        time.sleep(0.5)  # Wait a bit before the next update
        yield (
            "🔄 Pipeline is running...",
            log_capture.get_logs(),
            gr.update(),  # pdf_display
            gr.update(),  # llama_display
            gr.update(),  # agent2_display
            gr.update(),  # agent4_display
            gr.update(),  # time_display
            gr.update(),  # situation_display
            gr.update(),  # person_display
            gr.update(),  # environment_display
            gr.update(),  # challenge_display
            gr.update()   # activity_display
        )
    
    # The pipeline is finished, retrieve the final results
    thread.join()
    
    # Retrieve the result files
    patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
    pdf_html, llama_content, agent2_content, agent4_content = get_patient_files(patient_dir)
    time_content, situation_content, person_content, environment_content, challenge_content, activity_content = get_module_files(patient_dir)
    
    # Return the final results
    final_status = "✅ Pipeline executed successfully!" if os.path.exists(patient_dir) else "❌ Error during execution"
    
    yield (
        final_status,
        log_capture.get_logs(),
        pdf_html,
        llama_content,
        agent2_content,
        agent4_content,
        time_content,
        situation_content,
        person_content,
        environment_content,
        challenge_content,
        activity_content
    )

def load_patient_results(patient_name):
    """Loads results for an existing patient"""
    if not patient_name:
        return "", "", "", ""
    
    patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
    if not os.path.exists(patient_dir):
        return "<div style='text-align: center; padding: 20px; color: #dc3545;'>Patient not found</div>", "Patient not found", "", ""
    
    return get_patient_files(patient_dir)

def load_module_results(patient_name):
    """Loads Module results for an existing patient"""
    if not patient_name:
        return "", "", "", "", "", ""
    
    patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
    if not os.path.exists(patient_dir):
        return "Patient not found", "", "", "", "", ""
    
    return get_module_files(patient_dir)

def get_example_json():
    """Returns a JSON example for planning with all required fields"""
    return json.dumps({
        "nom": "P1",
        "dateNaissance": "10-02-2019",
        "classe": "CP1",
        "challenges": ["Frustration", "Faible estime de soi", "Stress chronique"],
        "planningEcole": [
            {
                "annee": 2025,
                "mois": "mai",  
                "jour": 14,
                "heureDebut": "08:00",
                "heureFin": "10:00",
                "matiere": "Math",
                "nomEnseignant": "Mathieu",
                "enClasse": "True"
            }
        ]
    }, indent=2, ensure_ascii=False)


def list_ground_truth_files() -> List[str]:
    """Lists all ground truth files"""
    ground_truth_dir = os.path.join(GRADIO_OUTPUT_DIR, "ground_truth")
    files = []
    if os.path.exists(ground_truth_dir):
        for item in os.listdir(ground_truth_dir):
            if item.endswith('.json'):
                files.append(item)
    return sorted(files)

def get_aggregated_file_path(patient_name: str) -> str:
    """Gets the path to the aggregated JSON file for a patient"""
    patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
    return os.path.join(patient_dir, "aggregated_results.json")


def create_metrics_table(evaluation_result, patient_name: str, model_name: str = "Mistral") -> str:
    """Creates an elegant HTML table for evaluation metrics"""
    
    print(f"🔍 Debug - Evaluation result type: {type(evaluation_result)}")
    print(f"🔍 Debug - Evaluation result content: {evaluation_result}")
    
    if not evaluation_result:
        return "<div style='padding: 20px; text-align: center; color: #dc3545;'><h3>❌ No evaluation metrics found</h3></div>"
    
    # Handle tuple format (metrics_dict, other_data)
    if isinstance(evaluation_result, tuple) and len(evaluation_result) >= 1:
        metrics = evaluation_result[0]  
    elif isinstance(evaluation_result, list) and len(evaluation_result) > 0:
        if isinstance(evaluation_result[0], tuple):
            metrics = evaluation_result[0][0]  
        else:
            metrics = evaluation_result[0]
    elif isinstance(evaluation_result, dict):
        metrics = evaluation_result
    else:
        return f"<div style='padding: 20px; text-align: center; color: #dc3545;'><h3>❌ Invalid evaluation result format</h3><p>Type: {type(evaluation_result)}</p></div>"
    
    print(f"🔍 Debug - Extracted metrics: {metrics}")
    print(f"🔍 Debug - Metrics type: {type(metrics)}")
    
    # Verify metrics
    expected_keys = ["precision", "recall", "f1", "onto_conf", "rel_halluc", "sub_halluc", "obj_halluc"]
    if not isinstance(metrics, dict) or not any(key in metrics for key in expected_keys):
        available_keys = list(metrics.keys()) if isinstance(metrics, dict) else "Not a dictionary"
        return f"""
        <div style='padding: 20px; text-align: center; color: #dc3545;'>
            <h3>❌ No valid metrics found</h3>
            <p>Available keys: {available_keys}</p>
            <p>Metrics type: {type(metrics)}</p>
        </div>
        """
    
    # Create the HTML table
    html_table = f"""
    <div style="max-width: 800px; margin: 20px auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px 12px 0 0; text-align: center;">
            <h2 style="margin: 0; font-size: 24px; font-weight: 600;">📊 Evaluation Results</h2>
        </div>
        
        <div style="background: #f8f9fa; padding: 15px; border-left: 4px solid #667eea;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #495057;">👤 Patient: {patient_name}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 600; color: #495057;">🤖 Model: {model_name}</span>
            </div>
        </div>
        
        <table style="width: 100%; border-collapse: collapse; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background: #343a40; color: white;">
                    <th style="padding: 15px; text-align: left; font-weight: 600;">📏 Metric</th>
                    <th style="padding: 15px; text-align: center; font-weight: 600;">📊 Value</th>
                    <th style="padding: 15px; text-align: center; font-weight: 600;">📈 Percentage</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Informations about metrics
    metric_info = {
        "precision": {"icon": "🎯", "name": "Precision", "description": "Accuracy of positive predictions"},
        "recall": {"icon": "🔍", "name": "Recall", "description": "Coverage of actual positives"},
        "f1": {"icon": "⚖️", "name": "F1-Score", "description": "Harmonic mean of precision and recall"},
        "onto_conf": {"icon": "🎭", "name": "Ontology Confidence", "description": "Confidence in ontology mapping"},
        "rel_halluc": {"icon": "🔗", "name": "Relation Hallucination", "description": "Rate of hallucinated relations"},
        "sub_halluc": {"icon": "👤", "name": "Subject Hallucination", "description": "Rate of hallucinated subjects"},
        "obj_halluc": {"icon": "🎯", "name": "Object Hallucination", "description": "Rate of hallucinated objects"}
    }
    
    
    row_count = 0
    for key, info in metric_info.items():
        if key in metrics:
            value = metrics[key]
            
            # Convert in a percentage
            try:
                percentage = f"{float(value) * 100:.1f}%"
                if key in ["precision", "recall", "f1", "onto_conf"]:
                    color = "#28a745" if float(value) >= 0.8 else "#ffc107" if float(value) >= 0.6 else "#dc3545"
                else:
                    color = "#28a745" if float(value) <= 0.1 else "#ffc107" if float(value) <= 0.3 else "#dc3545"
                
            except (ValueError, TypeError):
                percentage = "N/A"
                color = "#6c757d"
            
        
            bg_color = "#f8f9fa" if row_count % 2 == 0 else "#ffffff"
            
            html_table += f"""
                <tr style="background-color: {bg_color}; border-bottom: 1px solid #dee2e6;">
                    <td style="padding: 12px;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 18px; margin-right: 8px;">{info['icon']}</span>
                            <div>
                                <div style="font-weight: 600; color: #495057;">{info['name']}</div>
                                <div style="font-size: 12px; color: #6c757d; margin-top: 2px;">{info['description']}</div>
                            </div>
                        </div>
                    </td>
                    <td style="padding: 12px; text-align: center; font-weight: 600; color: #495057;">
                        {value}
                    </td>
                    <td style="padding: 12px; text-align: center;">
                        <span style="padding: 4px 8px; border-radius: 20px; background-color: {color}; color: white; font-weight: 600; font-size: 14px;">
                            {percentage}
                        </span>
                    </td>
                </tr>
            """
            row_count += 1
    
    html_table += """
            </tbody>
        </table>
    </div>
    """
    
    print(f"🔍 Debug - Generated HTML table length: {len(html_table)}")
    return html_table

def run_evaluation(patient_name: str, ground_truth_file: str):
    """Runs evaluation between aggregated results and ground truth"""
    if not patient_name or not ground_truth_file:
        return "❌ Error: Please select both patient and ground truth file", ""
    
    try:
        # Paths
        aggregated_path = get_aggregated_file_path(patient_name)
        ground_truth_path = os.path.join(GRADIO_OUTPUT_DIR, "ground_truth", ground_truth_file)
        
        # Check if files exist
        if not os.path.exists(aggregated_path):
            return f"❌ Error: Aggregated file not found for patient {patient_name}", ""
        
        if not os.path.exists(ground_truth_path):
            return f"❌ Error: Ground truth file not found: {ground_truth_file}", ""
        
        print(f"🔍 Starting evaluation for patient: {patient_name}")
        print(f"📄 System output: {aggregated_path}")
        print(f"📋 Ground truth: {ground_truth_path}")
        
        # Load JSON files
        with open(aggregated_path, 'r', encoding='utf-8') as f:
            system_output = json.load(f)
        
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        # Run evaluation
        evaluation_result = evaluation(system_output, ground_truth, patient_name)
        
        # Debug: Print the type and content of evaluation_result
        print(f"🔍 Debug - Evaluation result type: {type(evaluation_result)}")
        print(f"🔍 Debug - Evaluation result content: {evaluation_result}")
        
        # Save evaluation results
        patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
        eval_result_path = os.path.join(patient_dir, "evaluation_results.json")
        
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        
        print("✅ Evaluation completed successfully!")
        
        # Create metrics table
        metrics_table = create_metrics_table(evaluation_result, patient_name)
        
        # Return both the table and the raw results
        raw_results = json.dumps(evaluation_result, indent=2, ensure_ascii=False)
        
        return metrics_table, raw_results
        
    except Exception as e:
        error_msg = f"❌ Error during evaluation: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, ""

def load_evaluation_results(patient_name: str) -> str:

    """Loads existing evaluation results for a patient"""
    if not patient_name:
        return "No patient selected"
    
    patient_dir = os.path.join(GRADIO_OUTPUT_DIR, patient_name)
    eval_path = os.path.join(patient_dir, "evaluation_results.json")
    
    if not os.path.exists(eval_path):
        return "No evaluation results found for this patient"
    
    return load_file_safely(eval_path)

def excel_to_json(excel_file_path, nom_patient, date_naissance, classe, challenges):
    """Convert an Excel file to JSON with patient information"""
    if not excel_file_path or not os.path.exists(excel_file_path):
        return json.dumps({
            "nom": nom_patient or "",
            "dateNaissance": date_naissance or "",
            "classe": classe or "",
            "challenges": [c.strip() for c in challenges.split(',') if c.strip()] if challenges else [],
            "planningEcole": []
        }, indent=2, ensure_ascii=False)
    
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path)
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Create the base JSON structure
        json_data = {
            "nom": nom_patient or "",
            "dateNaissance": date_naissance or "",
            "classe": classe or "",
            "challenges": [c.strip() for c in challenges.split(',') if c.strip()] if challenges else [],
            "planningEcole": []
        }
        
        # Map columns according to your Excel format
        # Expected columns: Date, Start Time, End Time, Activity, Location, Teacher
        column_mapping = {
            'Date': ['Date', 'date', 'DATE'],
            'Heure de début': ['Heure de début', 'heure de début', 'Heure début', 'heureDebut'],
            'Heure de fin': ['Heure de fin', 'heure de fin', 'Heure fin', 'heureFin'], 
            'Activité': ['Activité', 'activité', 'Activite', 'matiere', 'Matière'],
            'Lieu': ['Lieu', 'lieu', 'LIEU', 'enClasse'],
            'Enseignant': ['Enseignant', 'enseignant', 'nomEnseignant', 'Professeur']
        }
        
        def find_column(df, possible_names):
            """Find the matching column in the DataFrame"""
            for col in df.columns:
                if col in possible_names:
                    return col
            return None
        
        # Find matching columns
        date_col = find_column(df, column_mapping['Date'])
        heure_debut_col = find_column(df, column_mapping['Heure de début'])
        heure_fin_col = find_column(df, column_mapping['Heure de fin'])
        activite_col = find_column(df, column_mapping['Activité'])
        lieu_col = find_column(df, column_mapping['Lieu'])
        enseignant_col = find_column(df, column_mapping['Enseignant'])
        
        # Convert each row into a planning entry
        for _, row in df.iterrows():
            try:
                # Parse the date
                date_str = str(row[date_col]) if date_col and pd.notna(row[date_col]) else ""
                
                # Try to parse different date formats
                annee, mois, jour = None, None, None
                if date_str:
                    try:
                        # Format DD/MM/YYYY
                        if '/' in date_str:
                            parts = date_str.split('/')
                            if len(parts) == 3:
                                jour = int(parts[0])
                                mois = int(parts[1])
                                annee = int(parts[2])
                        # Format YYYY-MM-DD
                        elif '-' in date_str:
                            parts = date_str.split('-')
                            if len(parts) == 3:
                                annee = int(parts[0])
                                mois = int(parts[1])
                                jour = int(parts[2])
                    except (ValueError, IndexError):
                        # If parsing fails, use default values
                        annee, mois, jour = 2025, 6, 20
                
                # Convert numeric month to name
                mois_noms = {
                    1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril',
                    5: 'mai', 6: 'juin', 7: 'juillet', 8: 'août',
                    9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'décembre'
                }
                mois_nom = mois_noms.get(mois, 'juin') if mois else 'juin'
                
                # Create the planning entry
                entry = {
                    "annee": annee or 2025,
                    "mois": mois_nom,
                    "jour": jour or 20,
                    "heureDebut": str(row[heure_debut_col]) if heure_debut_col and pd.notna(row[heure_debut_col]) else "08:00",
                    "heureFin": str(row[heure_fin_col]) if heure_fin_col and pd.notna(row[heure_fin_col]) else "09:00",
                    "matiere": str(row[activite_col]) if activite_col and pd.notna(row[activite_col]) else "Activité",
                    "nomEnseignant": str(row[enseignant_col]) if enseignant_col and pd.notna(row[enseignant_col]) else "Enseignant",
                    "enClasse": "True" if lieu_col and str(row[lieu_col]).upper() in ['SCHOOL', 'CLASSE', 'TRUE', 'ECOLE'] else "False"
                }
                
                json_data["planningEcole"].append(entry)
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        return json.dumps(json_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        # Return JSON with patient info even in case of error
        return json.dumps({
            "nom": nom_patient or "",
            "dateNaissance": date_naissance or "",
            "classe": classe or "",
            "challenges": [c.strip() for c in challenges.split(',') if c.strip()] if challenges else [],
            "planningEcole": []
        }, indent=2, ensure_ascii=False)
    
def create_example_excel():
    """Creates a sample Excel file and returns the corresponding JSON"""
    # Create a sample DataFrame with the new format
    example_data = {
        'Date': ['20/06/2025', '21/06/2025', '22/06/2025', '23/06/2025', '24/06/2025'],
        'Heure de début': ['08:30', '09:30', '14:00', '08:00', '10:30'],
        'Heure de fin': ['10:00', '11:00', '15:30', '09:30', '12:00'],
        'Activité': ['MATHS', 'DESSIN', 'Sciences', 'Histoire', 'Géographie'],
        'Lieu': ['SCHOOL', 'SCHOOL', 'MAISON', 'SCHOOL', 'SCHOOL'],
        'Enseignant': ['ALEX', 'MONIQUE', 'Pierre', 'Marie', 'Jean']
    }
    
    df = pd.DataFrame(example_data)
    
    # Save the example
    example_path = os.path.join(GRADIO_OUTPUT_DIR, "example_planning.xlsx")
    df.to_excel(example_path, index=False)
    
    print(f"📊 Sample Excel file created: {example_path}")
    print(f"📊 Example DataFrame:\n{df}")
    
    # Create the corresponding JSON structure for preview
    example_json = {
        "nom": "P1",
        "dateNaissance": "10-02-2019",
        "classe": "CP1",
        "challenges": ["Frustration", "Faible estime de soi", "Stress chronique"],
        "planningEcole": []
    }
    
    # Convert the DataFrame into planning entries
    for _, row in df.iterrows():
        # Parse the date
        date_parts = row['Date'].split('/')
        jour = int(date_parts[0])
        mois = int(date_parts[1])
        annee = int(date_parts[2])
        
        # Convert numeric month to name
        mois_noms = {
            1: 'janvier', 2: 'février', 3: 'mars', 4: 'avril',
            5: 'mai', 6: 'juin', 7: 'juillet', 8: 'août',
            9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'décembre'
        }
        
        entry = {
            "annee": annee,
            "mois": mois_noms[mois],
            "jour": jour,
            "heureDebut": str(row['Heure de début']),
            "heureFin": str(row['Heure de fin']),
            "matiere": str(row['Activité']),
            "nomEnseignant": str(row['Enseignant']),
            "enClasse": "True" if row['Lieu'].upper() == 'SCHOOL' else "False"
        }
        example_json["planningEcole"].append(entry)
    
    return json.dumps(example_json, indent=2, ensure_ascii=False)

def update_json_preview(excel_file, nom_patient, date_naissance, classe, challenges):
    """Updates the JSON preview with patient info and planning data"""
    return excel_to_json(excel_file, nom_patient, date_naissance, classe, challenges)

def create_and_return_example_excel():
    """Creates and returns the path to the example Excel file for download"""
    try:
        # Create example data with multiple planning entries
        example_data = {
            'annee': [2025, 2025, 2025, 2025, 2025],
            'mois': ['mai', 'mai', 'mai', 'juin', 'juin'],
            'jour': [14, 15, 16, 17, 18],
            'heureDebut': ['08:00', '09:30', '14:00', '08:00', '10:30'],
            'heureFin': ['10:00', '11:00', '15:30', '09:30', '12:00'],
            'matiere': ['Math', 'Français', 'Sciences', 'Histoire', 'Géographie'],
            'nomEnseignant': ['Mathieu', 'Sophie', 'Pierre', 'Marie', 'Jean'],
            'enClasse': ['True', 'True', 'False', 'True', 'True']
        }
        
        df = pd.DataFrame(example_data)
        
        # Create the example file
        example_path = os.path.join(GRADIO_OUTPUT_DIR, "example_planning.xlsx")
        df.to_excel(example_path, index=False)
        
        print(f"📊 Example Excel file created: {example_path}")
        return example_path
        
    except Exception as e:
        print(f"❌ Error creating example Excel: {e}")
        return None
    

def create_interface():
    with gr.Blocks(title="PATCHES Ontology", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 PATCHES : knowledge Extraction Crew (PEC)
        
        This interface allows you to execute PEC for information extraction 
        from PDF and Json documents.
        """)
        
        with gr.Tab("🚀 New Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 📁 Data Configuration")
                    
                    # PDF file upload
                    pdf_input = gr.File(
                        label="📄 PDF File (ADHD Criteria)",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    
                    # Uploaded PDF preview
                    pdf_preview = gr.HTML(
                        label="📄 PDF Preview",
                        value="<div style='text-align: center; padding: 20px; color: #666;'>No file selected</div>"
                    )
                    
                    # Patient information section
                    gr.Markdown("### 👤 Patient Information")
                    
                    # Patient name
                    patient_input = gr.Textbox(
                        label="👤 Patient Name (nom)",
                        placeholder="Enter patient name...",
                        value=""
                    )
                    
                    # Date of birth
                    date_naissance_input = gr.Textbox(
                        label="📅 Date of Birth (dateNaissance)",
                        placeholder="DD-MM-YYYY (e.g., 10-02-2019)",
                        value=""
                    )
                    
                    # Class
                    classe_input = gr.Textbox(
                        label="🎓 Class (classe)",
                        placeholder="Enter class (e.g., CP1, CE1, etc.)",
                        value=""
                    )
                    
                    # Challenges
                    challenges_input = gr.Textbox(
                        label="🎯 Challenges",
                        placeholder="Enter challenges separated by commas (e.g., Frustration, Faible estime de soi, Stress chronique)",
                        lines=3,
                        value=""
                    )
                    
                    # Excel planning file upload
                    gr.Markdown("### 📅 Planning Data (Excel File)")
                    
                    planning_excel_input = gr.File(
                        label="📊 Excel Planning File",
                        file_types=['.xlsx', '.xls'],
                        type="filepath"
                    )
                    
                    # JSON preview (lecture seule)
                    planning_json_preview = gr.Textbox(
                        label="📋 Complete JSON Preview (Patient Info + Planning)",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Fill in patient information and upload Excel file to see the complete JSON preview..."
                    )
                    
                    with gr.Row():
                        example_btn = gr.Button("📝 Load Example", size="sm")
                        download_example_btn = gr.Button("📥 Download Example Excel", size="sm")
                        update_json_btn = gr.Button("🔄 Update JSON Preview", size="sm")
                                        
                    # Execute button
                    run_btn = gr.Button(
                        "🚀 Launch Pipeline", 
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 📊 Status and Logs")
                    
                    # Status
                    status_output = gr.Textbox(
                        label="📈 Status",
                        interactive=False
                    )
                    
                    # Logs avec auto-scroll et style terminal
                    logs_output = gr.Textbox(
                        label="📋 Execution Logs (Real-time)",
                        lines=20,
                        max_lines=25,
                        interactive=False,
                        placeholder="The execution logs will appear here in real time...",
                        elem_classes=["terminal-logs"],
                        autoscroll=True
                    )
            
            # Profile results section in 4 parts
            gr.Markdown("## 📂 Profile Results")
            
            with gr.Row():
                with gr.Column():
                    pdf_display = gr.HTML(
                        label="📄 Analyzed PDF Document",
                        value="<div style='text-align: center; padding: 20px; color: #666;'>Results will appear after pipeline execution</div>"
                    )
                
                with gr.Column():
                    llama_display = gr.Textbox(
                        label="🔍 Parsing Results (Llama Parser)",
                        lines=20,
                        max_lines=25,
                        interactive=False,
                        placeholder="Llama parsing results will appear here..."
                    )
            
            with gr.Row():
                with gr.Column():
                    agent2_display = gr.Textbox(
                        label="🧠 Profile Interpretation Agent ",
                        lines=20,
                        max_lines=25,
                        interactive=False,
                        placeholder="Interpretation results will appear here..."
                    )
                
                with gr.Column():
                    agent4_display = gr.Textbox(
                        label="📊 Profile Extraction Agent   - Final Result (JSON)",
                        lines=20,
                        max_lines=25,
                        interactive=False,
                        placeholder="Final JSON results will appear here..."
                    )
            
            # Module results section in 2x3 grid
            gr.Markdown("## 🔧 Module Results")
            
            with gr.Row():
                with gr.Column():
                    time_display = gr.Textbox(
                        label="⏰ Module Extraction Agent - Time",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Time results will appear here..."
                    )
                
                with gr.Column():
                    situation_display = gr.Textbox(
                        label="📍 Module Extraction Agent - Situation",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Situation results will appear here..."
                    )
                
                with gr.Column():
                    person_display = gr.Textbox(
                        label="👤 Module Extraction Agent - Person",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Person results will appear here..."
                    )
            
            with gr.Row():
                with gr.Column():
                    environment_display = gr.Textbox(
                        label="🌍 Module Extraction Agent - Environment",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Environment results will appear here..."
                    )
                
                with gr.Column():
                    challenge_display = gr.Textbox(
                        label="🎯 Module Extraction Agent - Challenge",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Challenge results will appear here..."
                    )
                
                with gr.Column():
                    activity_display = gr.Textbox(
                        label="🎮 Module Extraction Agent - Activity",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        placeholder="Activity results will appear here..."
                    )
        
        with gr.Tab("📚 Profile History"):
            gr.Markdown("## 🔍 View Previous Profile Analyses")
            
            with gr.Row():
                with gr.Column(scale=1):
                    patients_dropdown = gr.Dropdown(
                        label="Select a Patient",
                        choices=list_processed_patients(),
                        interactive=True
                    )
                    
                    #refresh_patients_btn = gr.Button("🔄 Refresh List", size="sm")
                
                with gr.Column(scale=3):
                    pass
            
            # Historical Profile results display
            gr.Markdown("### Profile Results")
            
            with gr.Row():
                with gr.Column():
                    hist_pdf_display = gr.HTML(
                        label="📄 PDF Document",
                        value="<div style='text-align: center; padding: 20px; color: #666;'>Select a patient to view results</div>"
                    )
                
                with gr.Column():
                    hist_llama_display = gr.Textbox(
                        label="🔍 Parsing Results (Llama Parser)",
                        lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    hist_agent2_display = gr.Textbox(
                        label="🧠 Profile Interpretation Agent ",
                        lines=15,
                        interactive=False
                    )
                
                with gr.Column():
                    hist_agent4_display = gr.Textbox(
                        label="📊 Profile Extraction Agent   - Final Result (JSON)",
                        lines=15,
                        interactive=False
                    )
        
        with gr.Tab("🔧 Module History"):
            gr.Markdown("## 🔍 View Previous Module Analyses")
            
            with gr.Row():
                with gr.Column(scale=1):
                    module_patients_dropdown = gr.Dropdown(
                        label="Select a Patient",
                        choices=list_processed_patients(),
                        interactive=True
                    )
                    
                    #refresh_module_patients_btn = gr.Button("🔄 Refresh List", size="sm")
                
                with gr.Column(scale=3):
                    pass
            
            # Historical Module results display in 2x3 grid
            gr.Markdown("### Module Results")
            
            with gr.Row():
                with gr.Column():
                    hist_time_display = gr.Textbox(
                        label="⏰ Module Extraction Agent - Time",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to view results"
                    )
                
                with gr.Column():
                    hist_situation_display = gr.Textbox(
                        label="📍 Module Extraction Agent - Situation",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to view results"
                    )
                
                with gr.Column():
                    hist_person_display = gr.Textbox(
                        label="👤 Module Extraction Agent - Person",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to view results"
                    )
            
            with gr.Row():
                with gr.Column():
                    hist_environment_display = gr.Textbox(
                        label="🌍 Module Extraction Agent - Environment",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to view results"
                    )
                
                with gr.Column():
                    hist_challenge_display = gr.Textbox(
                        label="🎯 Module Extraction Agent - Challenge",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to view results"
                    )
                
                with gr.Column():
                    hist_activity_display = gr.Textbox(
                        label="🎮 Module Extraction Agent - Activity",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to view results"
                    )
        with gr.Tab("📊 Evaluation"):
            gr.Markdown("## 🔍 Evaluate System Output Against Ground Truth")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Configuration")
                    
                    # Patient selection
                    eval_patients_dropdown = gr.Dropdown(
                        label="👤 Select Patient",
                        choices=list_processed_patients(),
                        interactive=True
                    )
                    
                    # Ground truth file selection
                    ground_truth_dropdown = gr.Dropdown(
                        label="📋 Select Ground Truth File",
                        choices=list_ground_truth_files(),
                        interactive=True
                    )
                    
                    # # Refresh buttons
                    # with gr.Row():
                    #     refresh_eval_patients_btn = gr.Button("🔄 Refresh Patients", size="sm")
                    #     refresh_ground_truth_btn = gr.Button("🔄 Refresh Ground Truth", size="sm")
                    
                    # Patient name input
                    eval_patient_name = gr.Textbox(
                        label="✏️ Patient Name for Evaluation",
                        placeholder="Enter patient name...",
                        value=""
                    )
                    
                    # Evaluate button
                    evaluate_btn = gr.Button(
                        "🚀 Run Evaluation", 
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Evaluation Results")
                    
                    # Evaluation metrics table (HTML)
                    evaluation_metrics = gr.HTML(
                        label="📊 Evaluation Metrics",
                        value="<div style='text-align: center; padding: 40px; color: #666; background: #f8f9fa; border-radius: 10px;'>Evaluation results will appear here after running evaluation...</div>"
                    )
                    
                    # Raw evaluation results (collapsible)
                    with gr.Accordion("📋 Raw Evaluation Data", open=False):
                        evaluation_raw = gr.Textbox(
                            label="Raw Results (JSON)",
                            lines=20,
                            max_lines=25,
                            interactive=False,
                            placeholder="Raw evaluation data will appear here..."
                        )
            
            # File preview sections
            gr.Markdown("### 📄 File Previews")
            
            with gr.Row():
                with gr.Column():
                    aggregated_preview = gr.Textbox(
                        label="🔗 Aggregated System Output",
                        lines=15,
                        interactive=False,
                        placeholder="Select a patient to preview aggregated results..."
                    )
                
                with gr.Column():
                    ground_truth_preview = gr.Textbox(
                        label="📋 Ground Truth Data",
                        lines=15,
                        interactive=False,
                        placeholder="Select a ground truth file to preview..."
                    )
        
        # Events for "New Analysis" tab
        pdf_input.change(
            fn=create_pdf_preview,
            inputs=pdf_input,
            outputs=pdf_preview
        )

        # Auto-update JSON when any field changes
        planning_excel_input.change(
            fn=update_json_preview,
            inputs=[planning_excel_input, patient_input, date_naissance_input, classe_input, challenges_input],
            outputs=planning_json_preview
        )

        patient_input.change(
            fn=update_json_preview,
            inputs=[planning_excel_input, patient_input, date_naissance_input, classe_input, challenges_input],
            outputs=planning_json_preview
        )

        date_naissance_input.change(
            fn=update_json_preview,
            inputs=[planning_excel_input, patient_input, date_naissance_input, classe_input, challenges_input],
            outputs=planning_json_preview
        )

        classe_input.change(
            fn=update_json_preview,
            inputs=[planning_excel_input, patient_input, date_naissance_input, classe_input, challenges_input],
            outputs=planning_json_preview
        )

        challenges_input.change(
            fn=update_json_preview,
            inputs=[planning_excel_input, patient_input, date_naissance_input, classe_input, challenges_input],
            outputs=planning_json_preview
        )

        example_btn.click(
            fn=get_example_json,
            outputs=planning_json_preview
        )

        update_json_btn.click(
            fn=update_json_preview,
            inputs=[planning_excel_input, patient_input, date_naissance_input, classe_input, challenges_input],
            outputs=planning_json_preview
        )

        run_btn.click(
            fn=run_pipeline_with_realtime_updates,
            inputs=[pdf_input, patient_input, planning_excel_input, planning_json_preview],
            outputs=[status_output, logs_output, pdf_display, llama_display, agent2_display, agent4_display, 
                    time_display, situation_display, person_display, environment_display, challenge_display, activity_display]
        )

        # # Events for "Profile History" tab
        # refresh_patients_btn.click(
        #     fn=list_processed_patients,
        #     outputs=patients_dropdown
        # )

        
        patients_dropdown.change(
            fn=load_patient_results,
            inputs=patients_dropdown,
            outputs=[hist_pdf_display, hist_llama_display, hist_agent2_display, hist_agent4_display]
        )
        
        # # Events for "Module History" tab
        # refresh_module_patients_btn.click(
        #     fn= list_processed_patients,
        #     outputs=module_patients_dropdown
        # )

        module_patients_dropdown.change(
            fn=load_module_results,
            inputs=module_patients_dropdown,
            outputs=[hist_time_display, hist_situation_display, hist_person_display, 
                    hist_environment_display, hist_challenge_display, hist_activity_display]
        )

        # # Events for "Evaluation" tab
        # refresh_eval_patients_btn.click(
        #     fn= list_processed_patients,
        #     outputs=eval_patients_dropdown
        # )

        # refresh_ground_truth_btn.click(
        #     fn=list_ground_truth_files,
        #     outputs=ground_truth_dropdown
        # )

        # Preview aggregated file when patient is selected
        eval_patients_dropdown.change(
            fn=lambda patient: load_file_safely(get_aggregated_file_path(patient)) if patient else "",
            inputs=eval_patients_dropdown,
            outputs=aggregated_preview
        )

        # Copy patient name to input field
        eval_patients_dropdown.change(
            fn=lambda x: x if x else "",
            inputs=eval_patients_dropdown,
            outputs=eval_patient_name
        )

        # Preview ground truth file when selected
        ground_truth_dropdown.change(
            fn=lambda gt_file: load_file_safely(os.path.join(GRADIO_OUTPUT_DIR, "ground_truth", gt_file)) if gt_file else "",
            inputs=ground_truth_dropdown,
            outputs=ground_truth_preview
        )

        evaluate_btn.click(
            fn=run_evaluation,
            inputs=[eval_patient_name, ground_truth_dropdown],
            outputs=[evaluation_metrics, evaluation_raw]
        )

        download_example_btn.click(
            fn=create_and_return_example_excel,
            outputs=gr.File(label="📥 Example Excel File")
        )
    
    return demo

# Robust configuration for Docker
if __name__ == "__main__":
    print("🐳 Docker initialization...")

    # Create the interface
    demo = create_interface()  # Your existing function

    # Launch with Docker management
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    launch_gradio_docker(demo, port=port)