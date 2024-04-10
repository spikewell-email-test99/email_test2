using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace Test_Email_Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class TeamClassifierController : ControllerBase
    {
        private ILogger<TeamClassifierController> _logger;
        private dynamic? _vectorizer;
        private dynamic? _labelEncoder;
        private dynamic? _xgbModel;

        public TeamClassifierController(ILogger<TeamClassifierController> logger)
        {
            _logger = logger;

            // Initialize Python runtime
            PythonEngine.Initialize();

            // Load joblib files
            LoadJoblibFiles();
        }

        private void LoadJoblibFiles()
        {
            using (Py.GIL())
            {
                dynamic joblib = Py.Import("joblib");
                _vectorizer = joblib.load("MLModels/vectorizer.joblib");
                _labelEncoder = joblib.load("MLModels/label_encoder.joblib");
            }

            // Call the Python script to load the XGBoost model
            LoadXGBoostModel();
        }

        private void LoadXGBoostModel()
        {
            var pythonScript = @"
                                import joblib

                                # Load the XGBoost model from joblib file
                                with open('MLModels/xgb_classifier.joblib', 'rb') as f:
                                    xgb_model = joblib.load(f)
                                ";
            using (Py.GIL())
            {
                dynamic py = PythonEngine.ModuleFromString("__main__", pythonScript);
                _xgbModel = py.xgb_model;
            }
        }

        [HttpGet]
        public ActionResult<IEnumerable<string>> GetTopTeams([FromQuery] string description)
        {
            if (string.IsNullOrEmpty(description))
            {
                return BadRequest("Description cannot be empty");
            }

            // Transform input description using TF-IDF Vectorizer
            dynamic descriptionVectorized;
            using (Py.GIL())
            {
                descriptionVectorized = _vectorizer.transform(new[] { description });
            }

            // Get top 5 teams using the XGBoost model
            var top5TeamsIndices = PredictTopTeams(descriptionVectorized);

            // Decode indices to team names using Label Encoder
            var top5Teams = DecodeIndicesToTeamNames(top5TeamsIndices);

            return Ok(top5Teams);
        }

        private int[] PredictTopTeams(dynamic descriptionVectorized)
        {
            // Predict the top teams using the XGBoost model
            var prediction = _xgbModel.predict(descriptionVectorized);
            // Convert the prediction to array of integers representing the indices of the top teams
            return prediction.ToArray().Select.ToArray();
        }

        private List<string> DecodeIndicesToTeamNames(int[] indices)
        {
            // Decode indices to team names using Label Encoder
            var top5Teams = new List<string>();
            using (Py.GIL())
            {
                foreach (var index in indices)
                {
                    var teamName = _labelEncoder.inverse_transform(new[] { index }).First();
                    top5Teams.Add(teamName);
                }
            }
            return top5Teams;
        }
    }
}
