use anyhow::Result;
use candle::{DType, Device, Tensor};

pub struct BurnRateAnalyzer {
    time: Tensor,
    burn_data: Tensor,
    log_burn_data: Tensor,
    beta: Tensor,
    beta_std: Tensor,
    predictions: Tensor,
}

impl BurnRateAnalyzer {
    pub fn new(time_points: Vec<f32>, burn_data: Vec<f32>) -> Result<Self> {
        let device = Device::Cpu;

        // Convert input data to tensors
        let time = Tensor::new(&time_points[..], &device)?;
        let burn_data = Tensor::new(&burn_data[..], &device)?;
        let log_burn_data = burn_data.log()?;

        // Setup data matrices for regression
        let (x, y) = Self::setup_data(&time, &log_burn_data)?;

        // Fit the model
        let (beta, beta_std, predictions) = Self::fit_model(&x, &y)?;

        Ok(Self {
            time,
            burn_data,
            log_burn_data,
            beta,
            beta_std,
            predictions,
        })
    }

    fn setup_data(time: &Tensor, log_burn_data: &Tensor) -> Result<(Tensor, Tensor)> {
        // Prepare X matrix with time and ones column
        let n = time.dims()[0];
        let time_col = time.reshape((n, 1))?;
        let ones = Tensor::ones((n, 1), DType::F32, time.device())?;
        let x = Tensor::cat(&[&time_col, &ones], 1)?;

        // Prepare Y matrix
        let y = log_burn_data.reshape((n, 1))?;

        Ok((x, y))
    }

    fn solve_2x2_system(a: [[f32; 2]; 2], b: [f32; 2]) -> [f32; 2] {
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let x1 = (b[0] * a[1][1] - b[1] * a[0][1]) / det;
        let x2 = (a[0][0] * b[1] - a[1][0] * b[0]) / det;
        [x1, x2]
    }


    fn fit_model(x: &Tensor, y: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let n = x.dims()[0];

        // First, get time column and ones column using narrow
        let time_col = x.narrow(1, 0, 1)?.squeeze(1)?;  // First column (time)
        let ones = x.narrow(1, 1, 1)?.squeeze(1)?;      // Second column (ones)

        // Calculate necessary sums
        let time_sum = time_col.sum_all()?.to_scalar::<f32>()?;
        let time_sq_sum = time_col.mul(&time_col)?.sum_all()?.to_scalar::<f32>()?;
        let y_squeezed = y.squeeze(1)?;  // Remove the extra dimension from y
        let y_sum = y_squeezed.sum_all()?.to_scalar::<f32>()?;

        // Calculate x'y terms
        let xy_sum = time_col.mul(&y.squeeze(1)?)?.sum_all()?.to_scalar::<f32>()?;

        // Form X'X matrix
        let x_t_x = [[time_sq_sum, time_sum],
                     [time_sum, n as f32]];

        // Form X'y vector
        let x_t_y = [xy_sum, y_sum];

        // Solve system manually
        let beta_values = Self::solve_2x2_system(x_t_x, x_t_y);

        // Convert back to tensors
        let beta = Tensor::new(&beta_values, x.device())?.reshape((2, 1))?;

        // Compute predictions
        let predictions = x.matmul(&beta)?;

        // Compute residuals and standard errors
        let residuals = y.sub(&predictions)?;
        let residuals_sq = residuals.mul(&residuals)?;
        let residual_sq_sum = residuals_sq.sum_all()?.to_scalar::<f32>()?;
        let sigma_squared = residual_sq_sum / ((n - 2) as f32);

        // Compute standard errors manually
        let var_b1 = sigma_squared / (time_sq_sum - time_sum * time_sum / (n as f32));
        let var_b0 = sigma_squared * (time_sq_sum) /
                     (n as f32 * (time_sq_sum - time_sum * time_sum / (n as f32)));

        let beta_std = Tensor::new(&[var_b1.sqrt(), var_b0.sqrt()], x.device())?.reshape((2, 1))?;

        Ok((beta, beta_std, predictions))
    }


    pub fn print_results(&self, conversion: f32) -> Result<()> {

        let beta_value = self.beta.get(0)?.reshape(())?.to_scalar::<f32>()?;
        let beta_std_value = self.beta_std.get(0)?.reshape(())?.to_scalar::<f32>()?;

        let rate = 100.0 * conversion * beta_value;
        let sigma = beta_value / beta_std_value;

        println!("{}% YoY rate of increase with {:.2}σ certainty",
                rate.round(), sigma);
        Ok(())
    }
}