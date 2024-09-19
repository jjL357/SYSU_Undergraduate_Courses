package com.example.demo.controller;

import com.example.demo.model.User;
import com.example.demo.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class RegisterController {

    @Autowired
    private UserService userService;

    @GetMapping("/register")
    public String registerForm(Model model) {
        model.addAttribute("user", new User());
        return "register";
    }

    @PostMapping("/register")
    public String registerSubmit(@ModelAttribute User user, Model model) {
        if (!userService.isUsernameUnique(user.getName())) {
            model.addAttribute("usernameExists", true);
            return "register";
        }

        userService.saveUser(user);
        User lastRegisteredUser = userService.getLastRegisteredUser();
        model.addAttribute("lastRegisteredUser", lastRegisteredUser);
        return "redirect:/registerSuccess";
    }

    @GetMapping("/registerSuccess")
    public String registerSuccess(Model model) {
        User lastRegisteredUser = userService.getLastRegisteredUser();
        model.addAttribute("lastRegisteredUser", lastRegisteredUser);
        return "registerSuccess";
    }
}

