package com.library.booksystem.dto.request;

import jakarta.validation.constraints.Size;
import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.FieldDefaults;

@Data
@NoArgsConstructor
@AllArgsConstructor
@FieldDefaults(level = AccessLevel.PRIVATE)
public class UserRequest {
    @Size(min = 4, message = "USERNAME_INVALID" )
    String username;
    String email;
    @Size(min = 8, message = "PASSWORD_INVALID")
    String password;
    String createAt;

}
