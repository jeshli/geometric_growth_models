use homoskedastic_regression::analyzer::BurnRateAnalyzer;
use anyhow::Result;

fn main() -> Result<()> {
    let time_points = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let burn_data = vec![17.0, 18.0, 32.0, 20.0, 20.0, 23.0, 28.0, 29.0, 52.0, 135.0];

    let analyzer = BurnRateAnalyzer::new(time_points, burn_data)?;
    analyzer.print_results(12.0)?;

    Ok(())
}