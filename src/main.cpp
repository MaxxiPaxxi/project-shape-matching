#include "computations.h"

#include <GLFW/glfw3.h>
#include <atomic>
#include <chrono>
#include <igl/file_dialog_open.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <random>
#define DATA 0

struct picks
{
    unsigned int k  = 0;
    unsigned int vi = 0;
    bool picked{false};
    deformations::mesh* object;
    Eigen::Vector3f bc;
    int mouse_x, mouse_y;
};

struct simulation_params_t
{
    float Rb              = 0.1f;
    float beta            = 0.8f;
    float Famplitude      = 10.f;
    float pick_force      = 10.f;
    float dt              = 0.1f;
    float tau             = 0.8f;
    float perturbation    = 0.3f;
    bool visualize_forces = true;
    bool show_triangles   = false;
    deformations::deformation_type type = deformations::deformation_type::linear;
    bool pause                    = false;
    unsigned int triangles        = 0;
};

int user_input_force = 0; // 0: none; 1: up; 2: down; 3: left; 4: right; 5: forward; 6: backward;
picks pick;
deformations::mesh object;
std::atomic<bool> object_loaded{false};
simulation_params_t sim_params;
std::atomic<bool> new_physics_update = false;


bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {

  if (key == '1') {
    std::cout << "Applying upward force" << std::endl;
    user_input_force = 1;
    return true;
  }
  if (key == '2') {
    std::cout << "Applying downward force" << std::endl;
    user_input_force = 2;
    return true;
  }
  if (key == '3') {
    std::cout << "Applying leftward force" << std::endl;
    user_input_force = 3;
    return true;
  }
  if (key == '4') {
    std::cout << "Applying rightward force" << std::endl;
    user_input_force = 4;
    return true;
  }
  if (key == '5') {
    std::cout << "Applying forward force" << std::endl;
    user_input_force = 5;
    return true;
  }
  if (key == '6') {
    std::cout << "Applying backward force" << std::endl;
    user_input_force = 6;
    return true;
  }
  if (key == '7') {
    std::cout << "Resetting forces" << std::endl;
    user_input_force = 0;
    return true;
  }

  return false;
}

bool mouse_down(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  using button_type = igl::opengl::glfw::Viewer::MouseButton;
        if (static_cast<button_type>(button) != button_type::Left)
            return false;

        if (modifier == GLFW_MOD_SHIFT)
        {
            double const x = static_cast<double>(viewer.current_mouse_x);
            double const y =
                viewer.core().viewport(3) - static_cast<double>(viewer.current_mouse_y);

            int fid;
            Eigen::Vector3f bc{};

            if (igl::unproject_onto_mesh(
                    Eigen::Vector2f(x, y),
                    viewer.core().view,
                    viewer.core().proj,
                    viewer.core().viewport,
                    object.V(),
                    object.F(),
                    fid,
                    bc))
            {
                auto const& F = object.F();
                Eigen::Vector3i const face{F(fid, 0), F(fid, 1), F(fid, 2)};
                unsigned int closest_vertex = face(0);

                if (bc(1) > bc(0) && bc(1) > bc(2))
                {
                    closest_vertex = face(1);
                }
                else if (bc(2) > bc(0) && bc(2) > bc(1))
                {
                    closest_vertex = face(2);
                }

                object.set_fixed(closest_vertex, !object.is_fixed(closest_vertex));
            }

            return true;
        }

        if (modifier == GLFW_MOD_CONTROL)
        {
            // Pick points
            int fid;
            double const x = static_cast<double>(viewer.current_mouse_x);
            double const y =
                viewer.core().viewport(3) - static_cast<double>(viewer.current_mouse_y);

            Eigen::Vector3f bc{};

            bool const hit = igl::unproject_onto_mesh(
                Eigen::Vector2f(x, y),
                viewer.core().view,
                viewer.core().proj,
                viewer.core().viewport,
                object.V(),
                object.F(),
                fid,
                bc);

            if (hit)
            {
                pick.picked = true;
                pick.k      = fid;
                pick.object = &object;
                pick.bc     = bc;

                pick.mouse_x = viewer.current_mouse_x;
                pick.mouse_y = viewer.current_mouse_y;

                auto const& F = pick.object->F();
                Eigen::Vector3i const face{F(pick.k, 0), F(pick.k, 1), F(pick.k, 2)};
                int closest_vertex = face(0);

                if (pick.bc(1) > pick.bc(0) && pick.bc(1) > pick.bc(2))
                {
                    closest_vertex = face(1);
                }
                else if (pick.bc(2) > pick.bc(0) && pick.bc(2) > pick.bc(1))
                {
                    closest_vertex = face(2);
                }

                pick.vi = closest_vertex;
            }

            return true;
        }

        return false;

}

bool mouse_up(igl::opengl::glfw::Viewer& viewer, int button, int modifier)
{
  using button_type = igl::opengl::glfw::Viewer::MouseButton;
  if (static_cast<button_type>(button) != button_type::Left)
    return false;

  if (pick.picked) { pick.picked = false;}

  return true;
}

auto simulate(igl::opengl::glfw::Viewer& viewer)
{
        if (!object_loaded)
            return;

        if (user_input_force != 0)
        {
            Eigen::Vector3d const geometric_center = object.V().colwise().mean();
            Eigen::Vector4d force4d{0., 0., 0., 1.};

            switch (user_input_force)
            {
                case 1: force4d(1) = 1.; break;
                case 2: force4d(1) = -1.; break;
                case 3: force4d(0) = -1.; break;
                case 4: force4d(0) = 1.; break;
                case 5: force4d(2) = 1.; break;
                case 6: force4d(2) = -1.; break;
            }

            force4d *= sim_params.Famplitude;
            force4d.w() = 1.0;

            // convert force to world space
            Eigen::Matrix4d const screen_to_world_transform =
                (viewer.core().proj * viewer.core().view).inverse().cast<double>();
            Eigen::Vector3d const force = (screen_to_world_transform * force4d).segment(0, 3);

            Eigen::MatrixXd const f = object.apply_force(geometric_center, force);

            if (sim_params.visualize_forces)
            {
                Eigen::VectorXd const s = f.rowwise().norm();
                viewer.data().set_data(s);
            }

            user_input_force = 0;
        }

        if (!sim_params.visualize_forces)
        {
            Eigen::VectorXd s(object.V().rows());
            s.setConstant(1.0);
            viewer.data().set_data(s);
        }

        object.set_type(sim_params.type);
        object.set_tau(sim_params.tau);
        object.set_rayleigh_beta(sim_params.Rb);
        object.set_beta(sim_params.beta);
        object.set_perturbation(sim_params.perturbation);

        if (sim_params.type == deformations::deformation_type::quadratic)
        {
            object.integrate_quadratic(sim_params.dt);
        }
        else
        {
            object.integrate(sim_params.dt);
        }

        viewer.data().V = object.V();

        new_physics_update = true;

}

bool draw(igl::opengl::glfw::Viewer& viewer)
{

        if (!object_loaded)
            return false;

        if (pick.picked)
        {
            // Eigen::Vector3d picked_vertex_position{
            //     pick.object->V()(pick.vi, 0),
            //     pick.object->V()(pick.vi, 1),
            //     pick.object->V()(pick.vi, 2)};

            double const x1 = static_cast<double>(pick.mouse_x);
            double const y1 = viewer.core().viewport(3) - static_cast<double>(pick.mouse_y);

            double const x2 = static_cast<double>(viewer.current_mouse_x);
            double const y2 =
                viewer.core().viewport(3) - static_cast<double>(viewer.current_mouse_y);

            Eigen::Vector3d const p1 = igl::unproject(
                                           Eigen::Vector3f(x1, y1, .5f),
                                           viewer.core().view,
                                           viewer.core().proj,
                                           viewer.core().viewport)
                                           .cast<double>();
            Eigen::Vector3d const p2 = igl::unproject(
                                           Eigen::Vector3f(x2, y2, .5f),
                                           viewer.core().view,
                                           viewer.core().proj,
                                           viewer.core().viewport)
                                           .cast<double>();

            Eigen::Vector3d d = (p2 - p1).normalized();
            object.forces().row(pick.vi) += d * static_cast<double>(sim_params.pick_force);

            pick.mouse_x = viewer.current_mouse_x;
            pick.mouse_y = viewer.current_mouse_y;
        }

        if (!sim_params.pause)
        {
            simulate(viewer);
        }

        // draw fixed points
        viewer.data().clear_points();
        Eigen::MatrixX3d P(object.count_fixed(), 3);
        Eigen::MatrixX3d C(object.count_fixed(), 3);
        for (unsigned int i = 0, j = 0; i < object.V().rows(); ++i)
        {
            if (!object.is_fixed(i))
                continue;
            P.block(j, 0, 1, 3) = object.V().block(i, 0, 1, 3);
            C.block(j, 0, 1, 3) = Eigen::RowVector3d(255., 0., 0.);
            ++j;
        }
        viewer.data().add_points(P, C);

        if (new_physics_update)
        {
            viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_POSITION;
            new_physics_update = false;
        }

        viewer.data().show_lines = sim_params.show_triangles;
        sim_params.triangles    = object.F().rows();

        return false;

}





int main(int argc, char* argv[])
{
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> rd(0.f, 1.f);

    igl::opengl::glfw::Viewer viewer;
    viewer.data().show_labels = true;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_window = [&]() {
        // Define position and sizes of ImGui Menu
        ImGui::SetNextWindowPos(ImVec2(10.f * menu.menu_scaling(), 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiCond_FirstUseEver);
        ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoSavedSettings);

        bool dirty = false;
	static int file;

        if (ImGui::CollapsingHeader("Load OBJ Triangle Mesh", ImGuiTreeNodeFlags_DefaultOpen))
        {
	  if (ImGui::RadioButton("No Model Loaded", &file, 0))
	    {
	      object_loaded = false;
	      viewer.data().clear();
	    }
	  if (ImGui::RadioButton("Bunny", &file , 1))
          {
            object.load("../objects/bunny.obj");
	    object_loaded = true;
	    dirty = true;
          }
          if (ImGui::RadioButton("Cube", &file , 2))
          {
            object.load("../objects/cube.obj");
	    object_loaded = true;
	    dirty = true;
          }
          if (ImGui::RadioButton("Head", &file , 3))
          {
	    object.load("../objects/decimated-max.obj");
	    object_loaded = true;
	    dirty = true;
	    
          }
	  if (ImGui::RadioButton("Arm", &file , 4))
          {
	    object.load("../objects/arm.obj");
	    object_loaded = true;
	    dirty = true;
          }
	  if (ImGui::RadioButton("Camel", &file , 5))
          {
	    object.load("../objects/camel.obj");
	    object_loaded = true;
	    dirty = true;
          }
        }
	if (ImGui::CollapsingHeader("Type of Deformation", ImGuiTreeNodeFlags_DefaultOpen))
	{
	  if (ImGui::RadioButton("Linear", (int*)(&sim_params.type), 0)) { sim_params.type = deformations::deformation_type::linear; };
	  if (ImGui::RadioButton("Quadratic", (int*)(&sim_params.type), 1)) { sim_params.type = deformations::deformation_type::quadratic; };
	}
	
	if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
	  if (ImGui::Button("Up")) {user_input_force = 1;};
	  if (ImGui::Button("Left")) {user_input_force = 3;};
	  ImGui::SameLine();
	  if (ImGui::Button("Right")) {user_input_force = 4;};
	  if (ImGui::Button("Down")) {user_input_force = 2;};
	}

        // Expose the same variable directly ...
        if (ImGui::CollapsingHeader("Integration", ImGuiTreeNodeFlags_OpenOnArrow))
        {
            ImGui::SliderFloat("Velocity Damping", &sim_params.Rb, 0.f, 10.f);
            ImGui::SliderFloat("Force Amplitude", &sim_params.Famplitude, 0.f, 1000.f);
            ImGui::SliderFloat("Picking Force", &sim_params.pick_force, 1.f, 100.f);
            ImGui::SliderFloat("Tau", &sim_params.tau, 0.f, 1.f);
            ImGui::SliderFloat("Beta", &sim_params.beta, 0.f, 1.f);
            ImGui::SliderFloat("Regularization Perturbation", &sim_params.perturbation, 0.f, 0.1f);
            ImGui::SliderFloat("Time step", &sim_params.dt, 0.001f, 1.0f);
        }

        if (DATA != 0) // Some more data if needed
	  {
	    if (ImGui::CollapsingHeader("Data", ImGuiTreeNodeFlags_OpenOnArrow))
	      {
		ImGui::Checkbox("Show triangles", &sim_params.show_triangles);
		ImGui::Checkbox("Pause", &sim_params.pause);
		ImGui::Text("Triangles: %d", sim_params.triangles);
	      }
	  }
        if (dirty)
        {
	  viewer.data().clear();
	  viewer.data().set_mesh(object.V(), object.F());
	  viewer.core().align_camera_center(object.V());
        }    

        ImGui::End();
    };

    viewer.core().is_animating = true;
    viewer.callback_mouse_down = &mouse_down; // Handling mouse events
    viewer.callback_mouse_up = &mouse_up; // Handling mouse events
    viewer.callback_key_pressed = &key_down; // Handling Keyboard events
    viewer.callback_pre_draw = &draw; // Draw
    viewer.launch(); // Launch
    return EXIT_SUCCESS;
}

