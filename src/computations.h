#include <Eigen/Core>
#include <string>

namespace deformations {
  enum class deformation_type { linear, quadratic };

  class mesh
  {
  public:
    Eigen::MatrixXd V() { return x_; }
    Eigen::MatrixXi F() { return F_; }

    Eigen::MatrixX3d forces() { return f_; }
    Eigen::MatrixX3d velocities() { return v_; }
    Eigen::MatrixX3d positions() { return x_; }

    bool load(std::string const& filepath);

    //void apply_gravity() { f_.rowwise() += Eigen::RowVector3d{0.0, -9.8, 0.0}; }
    Eigen::MatrixX3d apply_force(Eigen::Vector3d center, Eigen::Vector3d force);

    void integrate(double dt);
    void integrate_quadratic(double dt);

    void set_rayleigh_beta(double Rb) { Rb_ = Rb; }
    void set_rayleigh_alpha(double Ra) { Ra_ = Ra; }

    double rayleigh_beta() { return Rb_; }
    double rayleigh_alpha() { return Ra_; }

    void set_beta(double beta) { beta_ = beta; }
    double beta() { return beta_; }

    void set_tau(double tau) { tau_ = tau; }
    double tau() { return tau_; }

    void set_perturbation(double perturbation) { perturbation_ = perturbation; }
    double perturbation() { return perturbation_; }

    void set_type(deformation_type type_of_deformation) { type_ = type_of_deformation; }
    deformation_type type() { return type_;}

    void set_fixed(unsigned int vi, bool fixed = true) { fixed_(vi) = fixed; }
    bool is_fixed(unsigned int vi) { return fixed_(vi); }
    unsigned int count_fixed() { return fixed_.count(); }
    void clear_fixed() { fixed_.fill(false); }
    
  private:
    void reset();

    Eigen::MatrixXd V0_;
    Eigen::MatrixX3d Q_;
    Eigen::Matrix3d AqqInv_;
    Eigen::Matrix<double, Eigen::Dynamic, 9> Qquadratic_;
    Eigen::Matrix<double, 9, 9> AqqInvQuadratic_;
    Eigen::MatrixXd x_;
    Eigen::MatrixX3d v_;
    Eigen::Array<bool, Eigen::Dynamic, 1> fixed_;
    Eigen::MatrixX3d f_;
    Eigen::MatrixXi F_;
    double Rb_ = 0.0;
    double Ra_ = 0.0;
    double tau_ = 1.0;
    double beta_ = 0.0;
    double perturbation_ = 0.1;
    deformation_type type_;
  };
} // Namespace deformations
