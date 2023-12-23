using System.Globalization;
using System.Text;

namespace CSBPNeuralNetworkSample;

public class BPNeuralNetwork
{
    private readonly double _learnRate;
    private readonly double _momentum;
    private readonly int _layers;
    private readonly int[] _layerSpec;
    private readonly double[][] _output;
    private readonly double[][] _delta;
    private readonly double[][][] _weight;
    private readonly double[][][] _weightChange;

    public BPNeuralNetwork(double learnRate, double momentum, int layers, int[] layerSpec)
    {
        this._learnRate = learnRate;
        this._momentum = momentum;
        this._layers = layers;
        this._layerSpec = layerSpec;

        _output = new double[layers][];
        _delta = new double[layers][];
        _weight = new double[layers][][];
        _weightChange = new double[layers][][];

        for (int i = 0; i < layers; i++)
        {
            _output[i] = new double[layerSpec[i]];
            _delta[i] = new double[layerSpec[i]];
            _weight[i] = new double[layerSpec[i]][];
            _weightChange[i] = new double[layerSpec[i]][];

            for (int j = 0; j < layerSpec[i] && i > 0; j++)
            {
                _weight[i][j] = new double[layerSpec[i - 1] + 1];
                _weightChange[i][j] = new double[layerSpec[i - 1] + 1];

                for (int k = 0; k <= layerSpec[i - 1]; k++)
                {
                    _weight[i][j][k] = new Random().NextDouble() * 2 - 1;
                    _weightChange[i][j][k] = 0.0;
                }
            }
        }
    }

    private double TransferFunction(double input)
    {
        return 1 / (1 + Math.Exp(-input)); //sigmoid
    }

    private double TransferFunctionDerivative(double input)
    {
        double transferResult = TransferFunction(input);
        return (1.0 - transferResult) * transferResult; //sigmoid derivative
    }
    
    public double MeanSquareError(double[] target)
    {
        double mse = 0.0;
    
        for (int i = 0; i < _layerSpec[_layers - 1]; i++)
        {
            mse += Math.Pow((target[i] - _output[_layers - 1][i]), 2);
        }

        return mse / _layerSpec[_layers - 1];
    }
    
    public void SetNetWeights(string weightString)
    {
        NumberStyles style = NumberStyles.AllowDecimalPoint | NumberStyles.AllowLeadingSign;
        CultureInfo culture = CultureInfo.InvariantCulture; // independent from the current culture
        
        var weightStrings = weightString.Split(";");
        long nextStringValue = 0;

        for (int i = 1; i < _layers; ++i)
        {
            for (int j = 0; j < _layerSpec[i]; ++j)
            {
                for (int k = 0; k < _layerSpec[i - 1] + 1; ++k)
                {
                    _weight[i][j][k] = double.Parse(weightStrings[nextStringValue++],style,culture);
                }
            }
        }
    }
    
    public string GetNetWeights()
    {
        StringBuilder weightString = new StringBuilder("");

        for (int i = 1; i < _layers; i++)
        {
            for (int j = 0; j < _layerSpec[i]; j++)
            {
                for (int k = 0; k < _layerSpec[i - 1] + 1; k++)
                {
                    weightString.Append(_weight[i][j][k].ToString("0.##########", CultureInfo.InvariantCulture));
                    weightString.Append(";");
                }
            }
        }

        return weightString.ToString();
    }
    
    public double OutValue(int position)
    {
        return _output[_layers - 1][position];
    }    
 
    public void FeedForward(double[] input)
    {
        // Move input data to the first layer
        for (int i = 0; i < _layerSpec[0]; i++)
        {
            _output[0][i] = input[i];
        }

        // Calculate output for each layer
        for (int i = 0; i < _layers; i++)
        {
            for (int j = 0; j < _layerSpec[i] && i > 0; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < _layerSpec[i - 1]; k++)
                {
                    // Prev layer output modified by its weight
                    sum += _output[i - 1][k] * _weight[i][j][k];
                }

                // And the BIAS
                sum += _weight[i][j][_layerSpec[i - 1]];

                _output[i][j] = TransferFunction(sum);
            }
        }
    }
    
    public void BackPropagate(double[] input, double[] target)
    {
        FeedForward(input);

        // Calculate output differences
        for(int i = 0; i < _layerSpec[_layers - 1]; i++)
        {
            _delta[_layers - 1][i] =
                TransferFunctionDerivative(_output[_layers - 1][i]) *
                (target[i] - _output[_layers - 1][i]);
        }

        //Calculate differences for previous layers
        for (int i = _layers - 2; i > 0; i--)
        {
            for (int j = 0; j < _layerSpec[i]; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < _layerSpec[i + 1]; k++)
                {
                    sum += _delta[i + 1][k] * _weight[i + 1][k][j];
                }
                _delta[i][j] = TransferFunctionDerivative(_output[i][j]) * sum;
            }
        }
        
        //Add some momentum... if any :) ...to keep it moving!
        for(int i = 1 ; i < _layers ; i++)
        {
            for(int j = 0; j < _layerSpec[i]; j++) 
            {
                for(int k=0; k < _layerSpec[i - 1]; k++) 
                {
                    _weight[i][j][k] += _momentum * _weightChange[i][j][k];
                }
                //For the BIAS too
                _weight[i][j][_layerSpec[i - 1]] += _momentum * _weightChange[i][j][_layerSpec[i - 1]];
            }
        }

        // Update weights and weightChanges
        for(int i = 1; i < _layers; i++)
        {
            for(int j = 0; j < _layerSpec[i]; j++)
            {
                for(int k = 0; k < _layerSpec[i - 1]; k++)
                {
                    _weightChange[i][j][k] = _learnRate * _delta[i][j] * _output[i - 1][k];
                    _weight[i][j][k] += _weightChange[i][j][k];
                }

                // Update the bias too
                _weightChange[i][j][_layerSpec[i - 1]] = _learnRate * _delta[i][j];
                _weight[i][j][_layerSpec[i - 1]] += _weightChange[i][j][_layerSpec[i - 1]];
            }
        }
    }
    
}